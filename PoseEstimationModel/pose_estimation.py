from __future__ import annotations

import json
import colorsys
import os
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import re
import sys

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/tmp")

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes


# ---------- Config helpers ----------

# COCO 17 Connections: Pairs of keypoint indices to connect
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), # Legs/Hips
    (5, 11), (6, 12), (5, 6),                        # Torso
    (5, 7), (6, 8), (7, 9), (8, 10),                 # Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),          # Head
    (3, 5), (4, 6)                                   # Ears to Shoulders
]

def _random_vibrant_bgr(rng: random.Random) -> tuple[int, int, int]:
    # Use HSV to avoid dark/greyish random RGBs; OpenCV uses BGR.
    h = rng.random()
    s = 0.75 + 0.25 * rng.random()
    v = 0.75 + 0.25 * rng.random()
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))

@dataclass
class ModelPaths:
    config: Optional[Path] = None
    checkpoint: Optional[Path] = None
    weights: Optional[Path] = None


class Config:
    """YAML wrapper with path resolution and dynamic attribute access."""

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent
        self.path = Path(config_path) if config_path else self.base_dir / "config.yaml"
        with open(self.path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize mutable flags
        self.pose_save_video = self.cfg.get("models", {}).get("pose", {}).get("save_video", False)
        self.det_save_video = self.cfg.get("models", {}).get("detection", {}).get("save_video", False)

    def resolve(self, p: str | Path | None) -> Optional[Path]:
        if p is None: return None
        p = Path(p)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def detection_weight(self) -> ModelPaths:
        det = self.cfg["models"]["detection"]
        name = det.get("name", "")
        variant = det.get("variant") or "n"
        weights = det.get("weights", {})
        family = "yolo26" if name.startswith("yolo26") else "yolo"

        available = weights.get(family, {})
        w = available.get(variant)
        if w is None and available:
            # fallback to first available weight
            w = next(iter(available.values()))
        if w is None:
            # final fallback: default lightweight model
            w = "yolov8n.pt"

        resolved = self.resolve(w)
        # If resolved is None or doesn't exist, try any existing file from available
        if (resolved is None or not Path(resolved).exists()) and available:
            for candidate in available.values():
                cand_path = self.resolve(candidate)
                if cand_path and cand_path.exists():
                    resolved = cand_path
                    break
        if resolved is None or not Path(resolved).exists():
            raise FileNotFoundError(
                f"Detection weights not found for family='{family}' variant='{variant}'. "
                f"Checked: {resolved}. Available variants: {list(available.keys())}"
            )
        return ModelPaths(weights=resolved)

    def pose_paths(self) -> ModelPaths:
        pose = self.cfg["models"]["pose"]
        name = pose.get("name", "").lower()
        variant = pose.get("variant", "large")
        weights = pose.get("weights", {})
        
        if "vit" in name:
            block = weights.get("vitpose", {}).get(variant, {})
            cfg_path = self.resolve(block.get("config"))
            ckpt_path = self.resolve(block.get("checkpoint"))
            if cfg_path is None or ckpt_path is None or not cfg_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(
                    f"VitPose weights/config not found for variant='{variant}'. "
                    f"cfg={cfg_path}, ckpt={ckpt_path}"
                )
            return ModelPaths(config=cfg_path, checkpoint=ckpt_path)
        if "rtm" in name:
            block = weights.get("rtm", {}).get(variant, {})
            cfg_path = self.resolve(block.get("config"))
            ckpt_path = self.resolve(block.get("checkpoint"))
            if cfg_path is None or ckpt_path is None or not cfg_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(
                    f"RTMPose weights/config not found for variant='{variant}'. "
                    f"cfg={cfg_path}, ckpt={ckpt_path}"
                )
            return ModelPaths(config=cfg_path, checkpoint=ckpt_path)
        
        block = weights.get("yolo-pose", {})
        w = block.get(variant) or block.get(name) or name
        resolved = self.resolve(w)
        if (resolved is None or not resolved.exists()) and block:
            # fallback to any existing file in block
            for candidate in block.values():
                cand_path = self.resolve(candidate)
                if cand_path and cand_path.exists():
                    resolved = cand_path
                    break
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(
                f"YOLO-Pose weights not found for variant='{variant}'. Expected at: {resolved}. "
                f"Available variants: {list(block.keys())}"
            )
        return ModelPaths(weights=resolved)

    @property
    def det_cfg(self) -> Dict[str, Any]: return self.cfg["models"]["detection"]
    
    @property
    def pose_cfg(self) -> Dict[str, Any]: return self.cfg["models"]["pose"]

    def resolved_paths(self) -> Dict[str, Path]:
        raw_paths = self.cfg.get("paths", {})
        paths = {"input_video": self.resolve(raw_paths["input_video"])}
        pose_root = self.resolve(raw_paths.get("pose_output_dir") or "../pose_outputs")
        paths["pose_output_dir"] = pose_root
        paths["pose_video_dir"] = pose_root / "pose_vis"
        return paths

    @property
    def pose_json_suffix(self) -> str:
        return self.cfg.get("paths", {}).get("pose_json_suffix", ".json")

    @property
    def static_prefix(self) -> str:
        return self.cfg.get("paths", {}).get("static_prefix", "01_")

    def human_centric_filename(self, video_stem: str) -> str:
        """Generate human-centric filename: 01_ + 4 digits (from video stem if possible)."""
        digits = "".join(re.findall(r"\d+", video_stem))
        if digits:
            suffix = digits[-4:].zfill(4)
        else:
            # deterministic fallback so UI can predict the filename
            suffix = "0000"
        return f"{self.static_prefix}{suffix}{self.pose_json_suffix}"


# ---------- Detection ----------

class PersonDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        det_params = cfg.det_cfg
        mp = cfg.detection_weight()
        
        self.model = YOLO(str(mp.weights))
        self.device = det_params.get("device", "cpu")
        self.model.to(self.device)

        self.conf = float(det_params.get("confidence_threshold", 0.3))
        self.iou = float(det_params.get("iou_threshold", 0.45))
        self.classes = det_params.get("classes") 
        
        trk_cfg = det_params.get("tracking", {})
        high_thresh = float(trk_cfg.get("track_high_thresh", self.conf))
        
        self.bt_args = Namespace(
            track_high_thresh=high_thresh,
            track_low_thresh=float(trk_cfg.get("track_low_thresh", 0.1)),
            new_track_thresh=float(trk_cfg.get("new_track_thresh", high_thresh)),
            match_thresh=float(trk_cfg.get("match_thresh", 0.8)),
            track_buffer=int(trk_cfg.get("track_buffer", 30)),
            min_box_area=float(trk_cfg.get("min_box_area", 10)),
            mot20=bool(trk_cfg.get("mot20", False)),
            fuse_score=bool(trk_cfg.get("fuse_score", False)),
            gmc=False,
        )

    def run(self, video_path: Path) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError(f"Cannot open: {video_path}")

        tracker = BYTETracker(self.bt_args, frame_rate=cap.get(cv2.CAP_PROP_FPS))
        records = []
        frame_id = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Detection & Tracking") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break

                kwargs = {"conf": self.conf, "iou": self.iou, "verbose": False, "device": self.device}
                if self.classes: kwargs["classes"] = self.classes

                # model() returns a list, we index to get the Results object
                results_list = self.model(frame, **kwargs)
                results = results_list[0]  # Results object

                if results.boxes is not None:
                    boxes = results.boxes
                    # Ensure tracker receives CPU tensors
                    if hasattr(boxes, "cpu"):
                        boxes = boxes.cpu()
                else:
                    boxes = Boxes(torch.zeros((0, 6)), frame.shape[:2])

                # BYTETracker expects a Boxes-like object with .conf/.cls attributes
                tracks = tracker.update(boxes, frame)

                for t in tracks:  # t: [x1, y1, x2, y2, track_id, score, cls, det_idx]
                    if len(t) >= 5:
                        x1, y1, x2, y2 = t[:4]
                        track_id = t[4]
                        score = t[5] if len(t) > 5 else None
                        cls = t[6] if len(t) > 6 else None
                        records.append({
                            "frame_id": frame_id,
                            "person_id": int(track_id),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "score": float(score) if score is not None else None,
                            "cls": int(cls) if cls is not None and not np.isnan(cls) else None,
                        })
                
                frame_id += 1
                pbar.update(1)

        cap.release()
        return records


# ---------- Pose Estimator Base ----------

class PoseEstimatorBase:
    """Abstract interface for pose estimators."""
    def process(self, video_path: Path, detections: List[Dict[str, Any]], output_dir: Path, json_suffix: str, output_name: Optional[str] = None) -> Path:
        raise NotImplementedError


# ---------- MMPose Concrete Estimator ----------

class MMPoseTopDownEstimator(PoseEstimatorBase):
    def __init__(self, cfg: Config, model_paths: ModelPaths):
        from mmpose.apis import inference_topdown, init_model
        self.cfg = cfg
        self.model = init_model(str(model_paths.config), str(model_paths.checkpoint), device=cfg.pose_cfg.get("device", "cpu"))
        self.inference_topdown = inference_topdown
        self.conf_threshold = float(cfg.pose_cfg.get("confidence_threshold", 0.25))
        self.expected_kp = None  # determined on first valid frame
        rng = random.Random(os.urandom(16))
        # One dynamic color per run (same across all persons for this video).
        self._vis_color = _random_vibrant_bgr(rng)
        self._vis_line_thickness = int(cfg.pose_cfg.get("vis_line_thickness", 2))
        self._vis_kpt_radius = int(cfg.pose_cfg.get("vis_kpt_radius", 2))

    def _estimate(self, frame, bbox):
        results = self.inference_topdown(self.model, frame, [bbox])
        if not results:
            return None

        try:
            sample = results[0]  # we pass one bbox at a time
            inst = sample.pred_instances
            kpts = inst.keypoints  # shape: (num, K, 2)
            scores = inst.keypoint_scores  # shape: (num, K)
            if torch.is_tensor(kpts):
                kpts = kpts.cpu()
            if torch.is_tensor(scores):
                scores = scores.cpu()
            kpts_np = kpts[0] if len(kpts.shape) == 3 else kpts
            scores_np = scores[0] if len(scores.shape) == 2 else scores
            triplets = [[float(x), float(y), float(s)] for (x, y), s in zip(kpts_np, scores_np)]
            return {"keypoints": triplets, "mean": float(np.mean(scores_np))}
        except Exception:
            return None

    def process(
        self,
        video_path: Path,
        detections: List[Dict[str, Any]],
        output_dir: Path,
        json_suffix: str,
        output_name: Optional[str] = None,
        output_video_path: Optional[Path] = None,
    ) -> Path:
        cap = cv2.VideoCapture(str(video_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        sparta_json = defaultdict(dict)
        frame_map = defaultdict(list)
        for d in detections: frame_map[d["frame_id"]].append(d)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        if output_video_path and self.cfg.pose_save_video:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*self.cfg.pose_cfg.get("fourcc", "XVID")), fps, (w, h))

        frame_id = 0
        with tqdm(total=total, desc="Pose Estimation") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break
                for p in frame_map.get(frame_id, []):
                    pose = self._estimate(frame, p["bbox"])
                    if pose and pose["mean"] >= self.conf_threshold:
                        kp_count = len(pose["keypoints"])
                        if self.expected_kp is None:
                            self.expected_kp = kp_count
                        if kp_count != self.expected_kp:
                            continue
                        
                        flat = [c for trip in pose["keypoints"] for c in trip]
                        sparta_json[str(p["person_id"])][str(frame_id)] = {
                            "keypoints": flat,
                            "scores": float(pose.get("mean", 0.0))
                        }

                        # --- Visualization Block ---
                        if writer is not None:
                            kpts = pose["keypoints"]
                            min_kpt_conf = float(self.cfg.pose_cfg.get("min_keypoint_confidence", 0.3))
                            color = self._vis_color

                            # 1. Draw Skeleton Lines
                            for start_idx, end_idx in SKELETON_CONNECTIONS:
                                if start_idx < kp_count and end_idx < kp_count:
                                    x1, y1, s1 = kpts[start_idx]
                                    x2, y2, s2 = kpts[end_idx]
                                    if s1 >= min_kpt_conf and s2 >= min_kpt_conf:
                                        cv2.line(
                                            frame,
                                            (int(x1), int(y1)),
                                            (int(x2), int(y2)),
                                            color,
                                            self._vis_line_thickness,
                                            lineType=cv2.LINE_AA,
                                        )

                            # 2. Draw Keypoint Dots
                            for (x, y, s) in kpts:
                                if s >= min_kpt_conf:
                                    cv2.circle(
                                        frame,
                                        (int(x), int(y)),
                                        self._vis_kpt_radius,
                                        color,
                                        -1,
                                        lineType=cv2.LINE_AA,
                                    )
                
                if writer is not None:
                    writer.write(frame)
                frame_id += 1
                pbar.update(1)

        cap.release()
        if writer is not None: writer.release()
        fname = output_name or f"{video_path.stem}{json_suffix}"
        out = output_dir / fname
        with open(out, "w") as f: json.dump(sparta_json, f, indent=2)
        return out


# ---------- YOLO Pose Estimator ----------

class YoloPoseEstimator(PoseEstimatorBase):
    def __init__(self, cfg: Config, model_paths: ModelPaths):
        self.cfg = cfg
        self.model = YOLO(str(model_paths.weights))
        self.conf = float(cfg.pose_cfg.get("confidence_threshold", 0.25))
        self.conf_threshold = self.conf  # Alias for interface consistency
        self.min_kpt_conf = float(cfg.pose_cfg.get("min_keypoint_confidence", 0.3))
        self.expected_kp = None  # determined on first valid frame
        rng = random.Random(os.urandom(16))
        # One dynamic color per run (same across all persons for this video).
        self._vis_color = _random_vibrant_bgr(rng)
        self._vis_line_thickness = int(cfg.pose_cfg.get("vis_line_thickness", 1))
        self._vis_kpt_radius = int(cfg.pose_cfg.get("vis_kpt_radius", 2))

    def process(
        self,
        video_path: Path,
        detections: List[Dict[str, Any]],
        output_dir: Path,
        json_suffix: str,
        output_name: Optional[str] = None,
        output_video_path: Optional[Path] = None,
    ) -> Path:
        cap = cv2.VideoCapture(str(video_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        sparta_json = defaultdict(dict)
        frame_map = defaultdict(list)
        for d in detections: frame_map[d["frame_id"]].append(d)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        if output_video_path and self.cfg.pose_save_video:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*self.cfg.pose_cfg.get("fourcc", "XVID")), fps, (w, h))

        frame_id = 0
        with tqdm(total=total, desc="YOLO Pose") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break
                persons = frame_map.get(frame_id, [])
                for p in persons:
                    x1, y1, x2, y2 = map(int, p["bbox"])
                    frame_h, frame_w = frame.shape[:2]
                    # Clamp bbox to frame to avoid negative indices wrapping from the end.
                    x1 = max(0, min(frame_w - 1, x1))
                    y1 = max(0, min(frame_h - 1, y1))
                    x2 = max(0, min(frame_w, x2))
                    y2 = max(0, min(frame_h, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    # Ultralytics returns a list[Results] for inference on an array.
                    results = self.model(roi, conf=self.conf, verbose=False)
                    if isinstance(results, (list, tuple)):
                        if not results:
                            continue
                        result = results[0]
                    else:
                        result = results

                    kpt_obj = getattr(result, "keypoints", None)
                    if kpt_obj is None or getattr(kpt_obj, "xy", None) is None or getattr(kpt_obj, "conf", None) is None:
                        continue
                    if len(kpt_obj.xy) == 0:
                        continue

                    # If multiple poses are detected inside the ROI, pick the most confident.
                    best_i = 0
                    boxes = getattr(result, "boxes", None)
                    if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf) > 0:
                        try:
                            best_i = int(boxes.conf.argmax().item())
                        except Exception:
                            best_i = 0

                    kps = kpt_obj.xy[best_i].cpu().numpy()
                    confs = kpt_obj.conf[best_i].cpu().numpy()

                    if self.expected_kp is None:
                        self.expected_kp = kps.shape
                    if kps.shape != self.expected_kp:
                        continue

                    kps[:, 0] += x1
                    kps[:, 1] += y1

                    triplets = [[float(x), float(y), float(c)] for (x, y), c in zip(kps, confs)]
                    flat = [c for trip in triplets for c in trip]
                    sparta_json[str(p["person_id"])][str(frame_id)] = {
                        "keypoints": flat,
                        "scores": float(confs.mean()),
                    }

                    # --- Visualization Block ---
                    if writer is not None:
                        color = self._vis_color
                        # 1. Draw Skeleton Lines
                        for start_idx, end_idx in SKELETON_CONNECTIONS:
                            if start_idx < len(kps) and end_idx < len(kps):
                                if confs[start_idx] >= self.min_kpt_conf and confs[end_idx] >= self.min_kpt_conf:
                                    p1 = (int(kps[start_idx][0]), int(kps[start_idx][1]))
                                    p2 = (int(kps[end_idx][0]), int(kps[end_idx][1]))
                                    cv2.line(frame, p1, p2, color, self._vis_line_thickness, lineType=cv2.LINE_AA)

                        # 2. Draw Keypoint Dots
                        for (x, y), c in zip(kps, confs):
                            if c >= self.min_kpt_conf:
                                cv2.circle(
                                    frame,
                                    (int(x), int(y)),
                                    self._vis_kpt_radius,
                                    color,
                                    -1,
                                    lineType=cv2.LINE_AA,
                                )

                if writer is not None: writer.write(frame)
                frame_id += 1
                pbar.update(1)

        cap.release()
        if writer is not None: writer.release()
        fname = output_name or f"{video_path.stem}{json_suffix}"
        out = output_dir / fname
        with open(out, "w") as f: json.dump(sparta_json, f, indent=2)
        return out


# ---------- Pipeline ----------

# --- Update in pose_estimation.py ---

class PosePipeline:
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.paths = self.config.resolved_paths()
        self.last_anomaly_records: List[Dict[str, Any]] = []
        self.last_pose_json_path: Optional[Path] = None

    # Inside PosePipeline class in pose_estimation.py

    def run_live(self):
        """
        Generator that yields processed frames with detection, tracking, pose estimation,
        and real-time SPARTA anomaly detection all happening in parallel.
        Uses rolling keypoint buffers for each person to enable live SPARTA inference.
        """
        p = self.paths
        cap = cv2.VideoCapture(str(p["input_video"]))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {p['input_video']}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Initialize detector, tracker, and pose estimator once
        detector = PersonDetector(self.config)
        tracker = BYTETracker(detector.bt_args, frame_rate=fps)
        
        pose_paths = self.config.pose_paths()
        pose_name = self.config.pose_cfg.get("name", "").lower()
        if "yolo" in pose_name:
            estimator = YoloPoseEstimator(self.config, pose_paths)
        elif "vit" in pose_name or "rtm" in pose_name:
            estimator = MMPoseTopDownEstimator(self.config, pose_paths)
        else:
            estimator = MMPoseTopDownEstimator(self.config, pose_paths)
        
        # --- SPARTA Anomaly Detection Setup ---
        try:
            from utils.tokenizer import Tokenizer
            from models import SPARTA_C, SPARTA_F, SPARTA_H
            from utils.train_utils import CostumLoss
            
            sparta_cfg = self.config.cfg.get("models", {}).get("sparta", {})
            seg_len = int(sparta_cfg.get("seg_len", 24))
            num_kp = int(sparta_cfg.get("num_kp", 18))
            relative = bool(sparta_cfg.get("relative", True))
            device = self.config.pose_cfg.get("device", "cpu")
            
            sparta_branch = sparta_cfg.get("branch", "SPARTA_C")
            friendly_branch_map = {
                "Reconstruction Model": "SPARTA_C",
                "Future trajectory prediction model": "SPARTA_F",
                "Hybrid": "SPARTA_H",
            }
            sparta_branch = friendly_branch_map.get(sparta_branch, sparta_branch)
            
            ckpt_c = sparta_cfg.get("checkpoints", {}).get("sparta_c") or ""
            ckpt_f = sparta_cfg.get("checkpoints", {}).get("sparta_f") or ""
            ckpt_h_c = sparta_cfg.get("checkpoints", {}).get("sparta_h_c") or ckpt_c
            ckpt_h_f = sparta_cfg.get("checkpoints", {}).get("sparta_h_f") or ckpt_f
            th_c = float(sparta_cfg.get("checkpoints", {}).get("eer_threshold_c", 0.5))
            th_f = float(sparta_cfg.get("checkpoints", {}).get("eer_threshold_f", 0.5))
            th_h = float(sparta_cfg.get("checkpoints", {}).get("eer_threshold_h", max(th_c, th_f)))
            
            # Build SPARTA model
            expand_ratio = 2 if relative else 1
            input_dim = num_kp * 2
            num_heads = int(sparta_cfg.get("model_num_heads", 2))
            latent_dim = int(sparta_cfg.get("model_latent_dim", 64))
            dropout = float(sparta_cfg.get("dropout", 0.3))
            
            if sparta_branch == "SPARTA_C":
                sparta_model = SPARTA_C(input_dim * expand_ratio, num_heads, latent_dim, 4, 1000, device=device, dropout=dropout)
                if ckpt_c and Path(ckpt_c).exists():
                    ckpt = torch.load(ckpt_c, map_location=device)
                    sparta_model.load_state_dict(ckpt['state_dict'])
                    print(f"[SPARTA] Loaded SPARTA_C from {ckpt_c}")
                sparta_model.to(device)
                sparta_model.eval()
                sparta_threshold = th_c
                sparta_tokenizer = Tokenizer(Namespace(branch="SPARTA_C", device=device, relative=relative, token_config="t", traj=False, num_kp=num_kp))
                loss_func = CostumLoss("MSE", a=1, b=1, c=1, d=1)
            elif sparta_branch == "SPARTA_F":
                sparta_model = SPARTA_F(input_dim * expand_ratio, num_heads, latent_dim, 4, 1000, device=device)
                if ckpt_f and Path(ckpt_f).exists():
                    ckpt = torch.load(ckpt_f, map_location=device)
                    sparta_model.load_state_dict(ckpt['state_dict'])
                    print(f"[SPARTA] Loaded SPARTA_F from {ckpt_f}")
                sparta_model.to(device)
                sparta_model.eval()
                sparta_threshold = th_f
                sparta_tokenizer = Tokenizer(Namespace(branch="SPARTA_F", device=device, relative=relative, token_config="t", traj=False, num_kp=num_kp))
                loss_func = CostumLoss("MSE", a=1, b=1, c=1, d=1)
            elif sparta_branch == "SPARTA_H":
                sparta_model = SPARTA_H(input_dim * expand_ratio, num_heads, latent_dim, 4, 1000, device=device, dropout=dropout)
                if ckpt_h_c and Path(ckpt_h_c).exists():
                    ckpt = torch.load(ckpt_h_c, map_location=device)
                    sparta_model.CTD.load_state_dict(ckpt['state_dict'])
                    print(f"[SPARTA] Loaded SPARTA_H CTD from {ckpt_h_c}")
                if ckpt_h_f and Path(ckpt_h_f).exists():
                    ckpt = torch.load(ckpt_h_f, map_location=device)
                    sparta_model.FTD.load_state_dict(ckpt['state_dict'])
                    print(f"[SPARTA] Loaded SPARTA_H FTD from {ckpt_h_f}")
                sparta_model.to(device)
                sparta_model.eval()
                sparta_threshold = th_h
                sparta_tokenizer = Tokenizer(Namespace(branch="SPARTA_H", device=device, relative=relative, token_config="t", traj=False, num_kp=num_kp))
                loss_func = CostumLoss("MSE", a=1, b=1, c=1, d=1)
            else:
                sparta_model = None
                sparta_tokenizer = None
                sparta_threshold = 0.5
                loss_func = None
            
            sparta_enabled = sparta_model is not None
        except Exception as e:
            print(f"[WARNING] SPARTA loading failed: {e}. Proceeding without anomaly detection.")
            sparta_enabled = False
            sparta_model = None
            sparta_tokenizer = None
        
        frame_id = 0
        sparta_json = defaultdict(dict)
        self.last_anomaly_records = []
        self.last_pose_json_path = None
        
        # Rolling buffers: per-person keypoint sequences for SPARTA
        per_person_kpts = defaultdict(list)  # {person_id: [(frame_id, keypoints_array), ...]}
        per_person_anomaly = defaultdict(lambda: defaultdict(lambda: 0))  # {person_id: {frame_id: 0/1}}
        
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # --- STEP 1: Detection & Tracking (unified) ---
                det_kwargs = {"conf": detector.conf, "iou": detector.iou, "verbose": False, "device": detector.device}
                if detector.classes:
                    det_kwargs["classes"] = detector.classes
                
                results_list = detector.model(frame, **det_kwargs)
                if isinstance(results_list, list) and len(results_list) > 0:
                    results = results_list[0]
                else:
                    results = results_list
                
                # Extract boxes
                if hasattr(results, 'boxes') and results.boxes is not None:
                    boxes = results.boxes.cpu()
                else:
                    boxes = Boxes(torch.zeros((0, 6)), frame.shape[:2])
                
                # Update tracker with detected boxes
                tracks = tracker.update(boxes, frame)
                
                # --- STEP 2: Pose Estimation ---
                for t in tracks:
                    if len(t) >= 5:
                        x1, y1, x2, y2 = t[:4]
                        track_id = int(t[4])
                        
                        # Extract bbox and run pose estimation
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        
                        if isinstance(estimator, YoloPoseEstimator):
                            pose = self._estimate_yolo_pose(estimator, frame, bbox)
                        else:
                            pose = estimator._estimate(frame, bbox)
                        
                        if pose and pose.get("mean", 0) >= estimator.conf_threshold:
                            # Store keypoints
                            flat = [c for trip in pose["keypoints"] for c in trip]
                            sparta_json[str(track_id)][str(frame_id)] = {
                                "keypoints": flat,
                                "scores": float(pose["mean"])
                            }
                            
                            anomaly_score = 0.0
                            pred = 0
                            # --- STEP 3: SPARTA Anomaly Detection (rolling buffer) ---
                            if sparta_enabled and len(pose["keypoints"]) > 0:
                                kpts_array = np.array([k[:2] for k in pose["keypoints"]], dtype=np.float32)  # (num_kp, 2)
                                per_person_kpts[track_id].append((frame_id, kpts_array))
                                
                                # Keep only recent seg_len frames
                                if len(per_person_kpts[track_id]) > seg_len + 10:
                                    per_person_kpts[track_id] = per_person_kpts[track_id][-(seg_len + 5):]
                                
                                # Run SPARTA inference when buffer has at least seg_len frames
                                if len(per_person_kpts[track_id]) >= seg_len:
                                    # Prepare batch data from recent seg_len frames
                                    recent_frames = per_person_kpts[track_id][-seg_len:]
                                    batch_kpts = np.array([kpt_arr for _, kpt_arr in recent_frames], dtype=np.float32)  # (seg_len, num_kp, 2)
                                    
                                    # Handle relative coordinates if enabled
                                    if relative and len(batch_kpts) > 1:
                                        abs_kpts = batch_kpts.copy()
                                        rel_kpts = batch_kpts.copy()
                                        for t_ in range(1, len(batch_kpts)):
                                            rel_kpts[t_] = batch_kpts[t_] - batch_kpts[t_ - 1]
                                        batch_kpts = np.concatenate([abs_kpts, rel_kpts], axis=1)  # (seg_len, num_kp*2, 2)
                                    
                                    # Convert to tensor: (1, channels=2, seg_len, num_kp)
                                    kpts_tensor = torch.from_numpy(batch_kpts.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)
                                    
                                    try:
                                        with torch.no_grad():
                                            if sparta_branch == "SPARTA_C":
                                                recon = sparta_model.forward(kpts_tensor, kpts_tensor)
                                                loss = loss_func.calculate(kpts_tensor, recon)
                                            elif sparta_branch == "SPARTA_F":
                                                pred = sparta_model.forward(kpts_tensor, kpts_tensor)
                                                loss = loss_func.calculate(kpts_tensor, pred)
                                            else:  # SPARTA_H
                                                ctd_out, ftd_out = sparta_model.forward(kpts_tensor, kpts_tensor)
                                                loss_c = loss_func.calculate(kpts_tensor, ctd_out)
                                                loss_f = loss_func.calculate(kpts_tensor, ftd_out)
                                                loss = (loss_c + loss_f) * 0.5
                                            
                                            anomaly_score = float(loss.cpu().numpy())
                                            pred = 1 if anomaly_score > sparta_threshold else 0
                                            
                                            # Store prediction for the most recent frame in the buffer
                                            most_recent_fid = recent_frames[-1][0]
                                            per_person_anomaly[track_id][most_recent_fid] = pred
                                    except Exception as e:
                                        print(f"[SPARTA] Inference error for person {track_id}: {e}")
                            
                            # Save record for this person/frame
                            self.last_anomaly_records.append({
                                "frame_id": frame_id,
                                "person_id": track_id,
                                "anomaly_score": anomaly_score,
                                "anomaly_pred": int(pred),
                            })
                            
                            # --- STEP 4: Visualization with anomaly coloring ---
                            anomaly_pred = per_person_anomaly[track_id].get(frame_id, 0)
                            frame = self._draw_pose_on_frame(
                                frame,
                                pose,
                                estimator,
                                bbox=bbox,
                                track_id=track_id,
                                is_anomaly=(anomaly_pred == 1),
                            )
                
                # --- STEP 5: Yield processed frame to Streamlit ---
                yield frame, frame_id, total_frames
                frame_id += 1
                pbar.update(1)
        
        cap.release()
        
        # Save the final JSON after all frames processed
        p["pose_output_dir"].mkdir(parents=True, exist_ok=True)
        out_name = self.config.human_centric_filename(Path(p["input_video"]).stem)
        final_path = p["pose_output_dir"] / out_name
        with open(final_path, "w") as f:
            json.dump(sparta_json, f, indent=2)
        
        self.last_pose_json_path = final_path
        return final_path

    def _estimate_yolo_pose(self, estimator, frame, bbox):
        """Extract pose from YOLO model within a bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        frame_h, frame_w = frame.shape[:2]
        
        # Clamp bbox to frame
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w, x2))
        y2 = max(0, min(frame_h, y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Run YOLO pose on ROI
        results = estimator.model(roi, conf=estimator.conf, verbose=False)
        if isinstance(results, list):
            if not results:
                return None
            result = results[0]
        else:
            result = results
        
        kpt_obj = getattr(result, "keypoints", None)
        if kpt_obj is None or not hasattr(kpt_obj, 'xy') or kpt_obj.xy is None or len(kpt_obj.xy) == 0:
            return None
        
        # Get best pose (most confident)
        best_i = 0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and hasattr(boxes, "conf") and len(boxes.conf) > 0:
            try:
                best_i = int(boxes.conf.argmax().item())
            except Exception:
                best_i = 0
        
        kps = kpt_obj.xy[best_i].cpu().numpy()
        confs = kpt_obj.conf[best_i].cpu().numpy()
        
        # Offset keypoints back to original frame coordinates
        kps[:, 0] += x1
        kps[:, 1] += y1
        
        triplets = [[float(x), float(y), float(c)] for (x, y), c in zip(kps, confs)]
        return {"keypoints": triplets, "mean": float(confs.mean())}

    def _draw_pose_on_frame(self, frame, pose, estimator, bbox: Optional[list[float]] = None, track_id: Optional[int] = None, is_anomaly: bool = False):
        """Draw skeleton, keypoints, bounding box, and ID on frame. Red color if anomaly detected."""
        # Use red if anomaly, otherwise use estimator color
        if is_anomaly:
            color = (0, 0, 255)  # Red in BGR
        else:
            color = estimator._vis_color if hasattr(estimator, '_vis_color') else (0, 255, 0)
        
        kpts = pose["keypoints"]
        min_kpt_conf = estimator.min_kpt_conf if hasattr(estimator, 'min_kpt_conf') else 0.3
        line_thickness = max(1, estimator._vis_line_thickness if hasattr(estimator, '_vis_line_thickness') else 2)
        kpt_radius = estimator._vis_kpt_radius if hasattr(estimator, '_vis_kpt_radius') else 3
        
        # Draw skeleton lines
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(kpts) and end_idx < len(kpts):
                x1, y1, s1 = kpts[start_idx]
                x2, y2, s2 = kpts[end_idx]
                if s1 >= min_kpt_conf and s2 >= min_kpt_conf:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness, lineType=cv2.LINE_AA)
        
        # Draw keypoint dots
        for (x, y, s) in kpts:
            if s >= min_kpt_conf:
                cv2.circle(frame, (int(x), int(y)), kpt_radius, color, -1, lineType=cv2.LINE_AA)

        if bbox is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness, lineType=cv2.LINE_AA)
            if track_id is not None:
                label = f"ID:{track_id}"
                text_pos = (x1, max(y1 - 6, 12))
                cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, lineType=cv2.LINE_AA)
                
                # Add anomaly indicator if detected
                if is_anomaly:
                    anomaly_label = "ANOMALY"
                    anomaly_pos = (x1, max(y1 - 20, 12))
                    cv2.putText(frame, anomaly_label, anomaly_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, lineType=cv2.LINE_AA)
                    cv2.putText(frame, anomaly_label, anomaly_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        
        return frame

# ---------- Main ----------

if __name__ == "__main__":
    config = Config("config.yaml")
    p = config.resolved_paths()

    print("\n[STEP 1] Running person detection...")
    detector = PersonDetector(config)
    dets = detector.run(p["input_video"])

    if dets:
        print("\n[STEP 2] Running pose estimation...")
        paths_pose = config.pose_paths()
        estimator = MMPoseTopDownEstimator(config, paths_pose)
        
        # Disable video saving manually if needed
        config.pose_save_video = False 

        hc_name = config.human_centric_filename(Path(p["input_video"]).stem)
        final_json = estimator.process(p["input_video"], dets, p["pose_output_dir"], config.pose_json_suffix, output_name=hc_name)
        print(f"\n[SUCCESS] Saved to: {final_json}")
    else:
        print("\n[SKIP] No persons found in detection phase.")
