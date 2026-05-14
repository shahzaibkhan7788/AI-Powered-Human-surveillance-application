import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from pose_estimation import Config, PosePipeline

FRIENDLY_BRANCH_MAP = {
    "Reconstruction Model": "SPARTA_C",
    "Future trajectory prediction model": "SPARTA_F",
    "Hybrid": "SPARTA_H",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run detection + pose estimation + SPARTA anomaly detection in one pass.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to PoseEstimationModel config YAML.")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path.")
    parser.add_argument("--branch", type=str, default="SPARTA_C", help="SPARTA branch: SPARTA_C, SPARTA_F, SPARTA_H or friendly names.")
    parser.add_argument("--output_video", type=str, default="sparta_visualized.avi", help="Output visualization video path.")
    parser.add_argument("--output_csv", type=str, default="sparta_anomaly_results.csv", help="Output anomaly CSV path.")
    parser.add_argument("--output_plot", type=str, default="sparta_anomaly_plot.png", help="Output anomaly plot image path.")
    parser.add_argument("--det_variant", type=str, help="YOLO detection variant: n, s, m, l, x.")
    parser.add_argument("--pose_variant", type=str, help="Pose model variant (e.g. small, base, large).")
    parser.add_argument("--pose_name", type=str, help="Pose model family name: vitpose, rtm, yolo-pose.")
    parser.add_argument("--device", type=str, help="Device for detection/pose/SPARTA, e.g. cpu or cuda.")
    parser.add_argument("--sparta_ckpt_c", type=str, help="Override path to SPARTA_C checkpoint.")
    parser.add_argument("--sparta_ckpt_f", type=str, help="Override path to SPARTA_F checkpoint.")
    parser.add_argument("--anomaly_threshold_c", type=float, help="Override SPARTA_C anomaly threshold.")
    parser.add_argument("--anomaly_threshold_f", type=float, help="Override SPARTA_F anomaly threshold.")
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.video_path:
        cfg.cfg.setdefault("paths", {})["input_video"] = args.video_path
    if args.branch:
        branch = FRIENDLY_BRANCH_MAP.get(args.branch, args.branch)
        cfg.cfg.setdefault("models", {}).setdefault("sparta", {})["branch"] = branch
    if args.det_variant:
        cfg.cfg.setdefault("models", {}).setdefault("detection", {})["variant"] = args.det_variant
    if args.pose_variant:
        cfg.cfg.setdefault("models", {}).setdefault("pose", {})["variant"] = args.pose_variant
    if args.pose_name:
        cfg.cfg.setdefault("models", {}).setdefault("pose", {})["name"] = args.pose_name
    if args.device:
        cfg.cfg.setdefault("models", {}).setdefault("detection", {})["device"] = args.device
        cfg.cfg.setdefault("models", {}).setdefault("pose", {})["device"] = args.device
    if args.sparta_ckpt_c:
        cfg.cfg.setdefault("models", {}).setdefault("sparta", {}).setdefault("checkpoints", {})["sparta_c"] = args.sparta_ckpt_c
    if args.sparta_ckpt_f:
        cfg.cfg.setdefault("models", {}).setdefault("sparta", {}).setdefault("checkpoints", {})["sparta_f"] = args.sparta_ckpt_f
    if args.anomaly_threshold_c is not None:
        cfg.cfg.setdefault("models", {}).setdefault("sparta", {}).setdefault("checkpoints", {})["eer_threshold_c"] = float(args.anomaly_threshold_c)
    if args.anomaly_threshold_f is not None:
        cfg.cfg.setdefault("models", {}).setdefault("sparta", {}).setdefault("checkpoints", {})["eer_threshold_f"] = float(args.anomaly_threshold_f)


def write_csv(records: list[dict], output_csv: Path) -> None:
    if not records:
        print("[WARN] No anomaly records were generated. CSV will be empty.")
    df = pd.DataFrame(records)
    df = df.sort_values(["frame_id", "person_id"])
    df.to_csv(output_csv, index=False)
    print(f"Saved anomaly CSV to: {output_csv}")


def plot_anomaly_scores(records: list[dict], output_plot: Path) -> None:
    if not records:
        print("[WARN] No records to plot.")
        return
    df = pd.DataFrame(records)
    df["frame_id"] = df["frame_id"].astype(int)
    df["person_id"] = df["person_id"].astype(int)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 5))

    for person_id, group in df.groupby("person_id"):
        ax.plot(group["frame_id"], group["anomaly_score"], label=f"person_{person_id}", alpha=0.75)

    ax.set_title("SPARTA Anomaly Score over Frames")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Anomaly Score")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)
    print(f"Saved plot to: {output_plot}")


def run_pipeline(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    pipeline = PosePipeline(str(config_path))
    apply_overrides(pipeline.config, args)
    pipeline.paths = pipeline.config.resolved_paths()

    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_plot = Path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    frame_count = 0

    for frame, frame_id, total_frames in pipeline.run_live():
        if writer is None:
            h, w = frame.shape[:2]
            input_video = pipeline.paths["input_video"]
            cap = cv2.VideoCapture(str(input_video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

        writer.write(frame)
        frame_count += 1
        if frame_id % 100 == 0 and frame_id > 0:
            print(f"Progress: {frame_id}/{total_frames}")

    if writer is not None:
        writer.release()

    if pipeline.last_pose_json_path is not None:
        print(f"Pose JSON saved at: {pipeline.last_pose_json_path}")
    else:
        print("[WARN] Pose JSON path was not captured.")

    write_csv(pipeline.last_anomaly_records, output_csv)
    plot_anomaly_scores(pipeline.last_anomaly_records, output_plot)
    print(f"✅ End-to-end SPARTA inference complete. Video saved to: {output_video}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
