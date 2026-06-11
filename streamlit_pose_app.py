import streamlit as st
import yaml
import tempfile
import re
import random
import subprocess
import pandas as pd
import cv2
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

# Local pipeline imports
from PoseEstimationModel.pose_estimation import Config, PosePipeline


BASE_DIR = Path(__file__).parent
POSE_DIR = BASE_DIR / "PoseEstimationModel"
BASE_CONFIG_PATH = POSE_DIR / "config.yaml"
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "default.yaml"
UPLOAD_DIR = POSE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def load_yaml_cfg(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_cfg() -> Dict:
    return load_yaml_cfg(BASE_CONFIG_PATH)


def load_default_cfg() -> Dict:
    return load_yaml_cfg(DEFAULT_CONFIG_PATH)


BASE_CFG = load_base_cfg()
DEFAULT_CFG = load_default_cfg()
POSE_WEIGHTS = BASE_CFG["models"]["pose"].get("weights", {})
DET_WEIGHTS = BASE_CFG["models"]["detection"].get("weights", {})
SPARTA_CFG = BASE_CFG.get("models", {}).get("sparta", {})
SPARTA_CORE_KEYS = {"branch", "relative", "token_config", "num_kp", "seg_len", "model_num_heads", "model_latent_dim", "dropout"}
SPARTA_WEIGHTSETS = {
    key: value
    for key, value in SPARTA_CFG.items()
    if isinstance(value, dict) and ("ctd" in value or "ftd" in value)
}


def cfg_float(cfg: Dict, key: str, fallback: float | None = None) -> float | None:
    value = cfg.get(key, fallback)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def branch_to_display(branch: str | None) -> str:
    return {
        "SPARTA_C": "SPARTA-C",
        "SPARTA_F": "SPARTA-F",
        "SPARTA_H": "SPARTA-H",
    }.get(str(branch or "").strip(), "SPARTA-C")


def infer_weightset_from_default() -> str:
    default_paths = [
        str(DEFAULT_CFG.get("model_ckpt_dir") or ""),
        str(DEFAULT_CFG.get("model_ckpt_C") or ""),
        str(DEFAULT_CFG.get("model_ckpt_F") or ""),
    ]
    for name, block in SPARTA_WEIGHTSETS.items():
        candidates = [str(block.get("ctd") or ""), str(block.get("ftd") or "")]
        if any(path_value and path_value in candidates for path_value in default_paths):
            return name
    if SPARTA_WEIGHTSETS:
        return next(iter(SPARTA_WEIGHTSETS.keys()))
    return ""

def variants_for(family: str) -> Tuple[str, ...]:
    block = POSE_WEIGHTS.get(family, {})
    return tuple(block.keys())

POSE_FAMILIES = tuple(POSE_WEIGHTS.keys())
POSE_VARIANTS = {family: variants_for(family) for family in POSE_FAMILIES}
DET_FAMILIES = tuple(DET_WEIGHTS.keys()) or ("yolo",)
DET_VARIANTS_BY_FAMILY = {
    family: tuple(DET_WEIGHTS.get(family, {}).keys()) or ("n", "s", "m", "l", "x")
    for family in DET_FAMILIES
}

def save_upload(upload) -> Path:
    target = UPLOAD_DIR / upload.name
    with open(target, "wb") as f:
        f.write(upload.getbuffer())
    return target

def camera_available(index: int = 0) -> bool:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return bool(ok)


def select_score_x_column(df: pd.DataFrame) -> str:
    for candidate in ("timeline_index", "start_frame", "frame_id", "segment"):
        if candidate in df.columns:
            return candidate
    return df.columns[0]


def build_person_score_figure(df: pd.DataFrame, threshold: float | None = None):
    import matplotlib.pyplot as plt

    frame = df.copy()
    if "person_id" in frame.columns:
        frame["person_id"] = frame["person_id"].astype(str)
    else:
        frame["person_id"] = "unknown"
    x_col = select_score_x_column(frame)
    if "score" not in frame.columns:
        score_col = [c for c in frame.columns if c not in {"person_id", x_col}][-1]
        frame = frame.rename(columns={score_col: "score"})
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=[x_col, "score"]).sort_values(["person_id", x_col])

    plot_threshold = float(threshold) if threshold is not None else None
    max_score = float(frame["score"].max()) if not frame.empty else 0.0
    scale_factor = 1.0
    if max_score > 4.0:
        scale_factor = max_score / 4.0
        frame["score"] = frame["score"] / scale_factor
        if plot_threshold is not None:
            plot_threshold = plot_threshold / scale_factor

    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=120)
    for person_id, group in frame.groupby("person_id", sort=False):
        ax.plot(
            group[x_col],
            group["score"],
            linewidth=0.9,
            alpha=0.85,
            label=f"ID {person_id}",
        )
    if plot_threshold is not None:
        ax.axhline(plot_threshold, color="red", linestyle="--", linewidth=1.1, label=f"Threshold {float(threshold):.3f}")
    ax.set_title("Human-Centric Anomaly Scores Over Time", fontsize=12)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=9)
    ax.set_ylabel("Anomaly Score", fontsize=9)
    ax.set_ylim(0, 4.2)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.margins(x=0.01)
    ax.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.78)
    if scale_factor > 1.0:
        ax.text(0.01, 0.97, f"display scaled ÷ {scale_factor:.1f}", transform=ax.transAxes, fontsize=7, va="top", color="gray")
    fig.tight_layout(pad=0.8)
    return fig


def get_expected_name(stem: str, prefix: str) -> str:
    """Replicates the 01_xxxx logic from pose_estimation.py"""
    digits = re.findall(r'\d+', stem)
    if digits:
        full_digit_str = "".join(digits)
        suffix = full_digit_str[-4:] if len(full_digit_str) >= 4 else full_digit_str.zfill(4)
    else:
        suffix = "0000" # Fallback if no digits found
    return f"{prefix}{suffix}"

def build_run_config(
    base_cfg: Dict,
    pose_family: str,
    pose_variant: str,
    det_family: str,
    det_variant: str,
    device: str,
    video_path: Path,
    save_video: bool,
    sparta_branch: str,
    ckpt_c: str,
    ckpt_f: str,
    th_c: float | None,
    th_f: float | None,
    th_h: float | None,
) -> Dict:
    cfg = deepcopy(base_cfg)
    cfg["models"]["pose"]["name"] = pose_family
    cfg["models"]["pose"]["variant"] = pose_variant
    cfg["models"]["detection"]["name"] = det_family
    cfg["models"]["detection"]["variant"] = det_variant
    
    if device != "auto":
        cfg["models"]["pose"]["device"] = device
        cfg["models"]["detection"]["device"] = device

    cfg["models"]["pose"]["save_video"] = bool(save_video)
    sparta_cfg = cfg.setdefault("models", {}).setdefault("sparta", {})
    checkpoints = sparta_cfg.setdefault("checkpoints", {})
    sparta_cfg["branch"] = sparta_branch
    if ckpt_c:
        checkpoints["sparta_c"] = ckpt_c
        checkpoints["sparta_h_c"] = ckpt_c
    if ckpt_f:
        checkpoints["sparta_f"] = ckpt_f
        checkpoints["sparta_h_f"] = ckpt_f
    if th_c is not None:
        checkpoints["eer_threshold_c"] = float(th_c)
    if th_f is not None:
        checkpoints["eer_threshold_f"] = float(th_f)
    if th_h is not None:
        checkpoints["eer_threshold_h"] = float(th_h)
    elif th_c is not None or th_f is not None:
        checkpoints["eer_threshold_h"] = max(float(v) for v in (th_c, th_f) if v is not None)
    cfg["paths"]["input_video"] = str(video_path)
    cfg["paths"]["static_prefix"] = "01_"
    cfg["paths"]["pose_json_suffix"] = ".json"
    return cfg


def resolve_sparta_defaults(selected_weights: str) -> Tuple[str, str, float | None, float | None]:
    block = SPARTA_WEIGHTSETS.get(selected_weights, {})
    default_ctd = block.get("ctd", "")
    default_ftd = block.get("ftd", "")
    threshold_c = block.get("eer_threshold_c")
    threshold_f = block.get("eer_threshold_f")
    return default_ctd, default_ftd, threshold_c, threshold_f


def build_sparta_config(
    base_cfg: Dict,
    sparta_branch: str,
    ckpt_c: str,
    ckpt_f: str,
    th_c: float | None,
    th_f: float | None,
    th_h: float | None,
    pose_json_dir: Path,
    device: str,
    selected_weightset: str,
) -> Dict:
    sparta_cfg = {
        "mode": "test",
        "no_metrics": True,
        "save_results": True,
        "save_results_dir": str(Path(base_cfg["paths"].get("sparta_output_dir", "evaluation_results_sparta")).resolve()),
        "mask_root": None,
        "pose_path_test": str(pose_json_dir),
        "vid_path_test": None,
        "branch": sparta_branch,
        "relative": base_cfg.get("models", {}).get("sparta", {}).get("relative", True),
        "token_config": base_cfg.get("models", {}).get("sparta", {}).get("token_config", "t"),
        "num_kp": base_cfg.get("models", {}).get("sparta", {}).get("num_kp", 18),
        "seg_len": base_cfg.get("models", {}).get("sparta", {}).get("seg_len", 24),
        "model_num_heads": base_cfg.get("models", {}).get("sparta", {}).get("model_num_heads", 2),
        "model_latent_dim": base_cfg.get("models", {}).get("sparta", {}).get("model_latent_dim", 64),
        "dropout": base_cfg.get("models", {}).get("sparta", {}).get("dropout", 0.3),
        "batch_size": base_cfg.get("batch_size", 256) if isinstance(base_cfg.get("batch_size", 256), int) else 256,
        "device": device if device != "auto" else base_cfg.get("models", {}).get("pose", {}).get("device", "cpu"),
        "dataset": base_cfg.get("dataset", "corridor"),
        "weight_preset": selected_weightset,
    }
    friendly_branch_map = {
        "SPARTA_C": "SPARTA_C",
        "SPARTA_F": "SPARTA_F",
        "Hybrid": "SPARTA_H",
    }
    sparta_branch = friendly_branch_map.get(sparta_branch, sparta_branch)
    if sparta_branch == "SPARTA_H":
        sparta_cfg["model_ckpt_C"] = ckpt_c
        sparta_cfg["model_ckpt_F"] = ckpt_f
        sparta_cfg["eer_threshold_c"] = th_c
        sparta_cfg["eer_threshold_f"] = th_f
        sparta_cfg["eer_threshold_h"] = th_h
    else:
        sparta_cfg["model_ckpt_dir"] = ckpt_c if sparta_branch == "SPARTA_C" else ckpt_f
        if sparta_branch == "SPARTA_C":
            sparta_cfg["eer_threshold_c"] = th_c
        else:
            sparta_cfg["eer_threshold_f"] = th_f
    return sparta_cfg

def main():
    st.set_page_config(page_title="Human Surveillance Application", page_icon="🕺", layout="wide")
    st.markdown(
        """
        <style>
        :root{
          --brand:#3b82f6;
          --brand-2:#60a5fa;
          --brand-dark:#1d4ed8;
          --ink:#0f172a;
          --muted:#475569;
          --border:rgba(148,163,184,0.55);
        }

        /* Sharper, cleaner overall look */
        [data-testid="stAppViewContainer"]{
          background:
            radial-gradient(900px 420px at 12% 0%, rgba(59,130,246,0.22), transparent 62%),
            radial-gradient(760px 460px at 92% 12%, rgba(96,165,250,0.16), transparent 56%),
            linear-gradient(180deg, #ffffff 0%, #f8fafc 68%, #f1f5f9 100%);
        }
        [data-testid="stHeader"]{ background: transparent; }
	        section.main > div.block-container{
	          padding-top: 2.1rem;
	          padding-bottom: 2.4rem;
	        }
	        h1, h2, h3{ color: var(--ink); letter-spacing: -0.01em; }
	        [data-testid="stCaptionContainer"]{ color: rgba(71,85,105,0.95); }

	        /* Two-panel layout as clean cards */
	        div[data-testid="stHorizontalBlock"]{
	          gap: 1.25rem;
	        }
	        div[data-testid="column"]{
	          background: rgba(255,255,255,0.86);
	          border: 1px solid rgba(148,163,184,0.42);
	          border-radius: 16px;
	          padding: 0.95rem 0.95rem 0.55rem;
	          box-shadow: 0 14px 28px rgba(15,23,42,0.06);
	        }

	        /* Primary button -> light blue (not red) */
	        button[kind="primary"], div[data-testid="baseButton-primary"] button, button[data-testid="baseButton-primary"]{
	          background: linear-gradient(90deg, var(--brand-dark) 0%, var(--brand) 45%, var(--brand-2) 100%) !important;
	          border: 1px solid rgba(29,78,216,0.85) !important;
	          color: #ffffff !important;
	          border-radius: 14px;
	          height: 3.05rem;
	          font-weight: 750;
	          box-shadow: 0 12px 22px rgba(29,78,216,0.20) !important;
	        }
	        button[kind="primary"]:hover, div[data-testid="baseButton-primary"] button:hover, button[data-testid="baseButton-primary"]:hover{
	          filter: saturate(1.08) brightness(1.02);
	          box-shadow: 0 16px 28px rgba(29,78,216,0.26);
	          transform: translateY(-1px);
	        }
	        button[kind="primary"]:active, div[data-testid="baseButton-primary"] button:active, button[data-testid="baseButton-primary"]:active{
	          transform: translateY(0px);
	        }

	        .big-button button {width:100%; border-radius:12px; height:3rem; font-weight:700;}
	        .metric-card {padding:12px 16px; border-radius:12px; background:#0c111c0d; border:1px solid #e5e7eb;}

	        /* Sharper inputs (select, text, number) */
	        div[data-baseweb="select"] > div{
	          border-radius: 12px;
	          border: 1px solid var(--border);
	          background: rgba(255,255,255,0.92);
	        }
	        div[data-baseweb="select"] > div:focus-within{
	          border-color: var(--brand);
	          box-shadow: 0 0 0 3px rgba(59,130,246,0.14);
	        }
	        div[data-testid="stTextInput"] input,
	        div[data-testid="stNumberInput"] input{
	          border-radius: 12px !important;
	          border: 1px solid var(--border) !important;
	          background: rgba(255,255,255,0.92) !important;
	        }
	        div[data-testid="stTextInput"] input:focus,
	        div[data-testid="stNumberInput"] input:focus{
	          border-color: var(--brand) !important;
	          box-shadow: 0 0 0 3px rgba(59,130,246,0.14) !important;
	        }

	        /* Tabs accent */
	        div[data-testid="stTabs"] button[aria-selected="true"]{
	          color: var(--brand-dark) !important;
	        }

	        /* --- Video uploader: clean light-blue dropzone --- */
	        :root{
	          --upload-accent:var(--brand-2);
          --upload-accent-strong:var(--brand);
          --upload-bg:rgba(79, 141, 247, 0.06);
          --upload-border:rgba(79, 141, 247, 0.35);
          --upload-border-strong:rgba(79, 141, 247, 0.65);
        }

        div[data-testid="stFileUploader"], section[data-testid="stFileUploader"]{
          border:1px solid var(--upload-border);
          border-radius:14px;
          padding:0.75rem 0.85rem 0.9rem;
          background:linear-gradient(180deg, var(--upload-bg), rgba(255,255,255,0.0));
        }

        div[data-testid="stFileUploader"] label, section[data-testid="stFileUploader"] label{
          font-weight:600;
        }

        div[data-testid="stFileUploaderDropzone"], section[data-testid="stFileUploaderDropzone"]{
          border:2px dashed var(--upload-border-strong);
          border-radius:12px;
          background:rgba(255,255,255,0.88);
        }

        div[data-testid="stFileUploaderDropzone"]:hover, section[data-testid="stFileUploaderDropzone"]:hover{
          border-color:var(--upload-accent-strong);
          box-shadow:0 0 0 3px rgba(59,130,246,0.12);
        }

        div[data-testid="stFileUploaderDropzone"] svg, section[data-testid="stFileUploaderDropzone"] svg{
          color:var(--upload-accent-strong);
        }

        div[data-testid="stFileUploaderDropzone"] button, section[data-testid="stFileUploaderDropzone"] button{
          border:1px solid var(--upload-border-strong);
          color:var(--upload-accent-strong);
          background:rgba(255,255,255,0.95);
        }

        div[data-testid="stFileUploaderDropzone"] button:hover,
        section[data-testid="stFileUploaderDropzone"] button:hover{
          border-color:var(--upload-accent-strong);
          background:rgba(59,130,246,0.08);
        }

        div[data-testid="stFileUploader"] small, section[data-testid="stFileUploader"] small{
          color:rgba(15, 23, 42, 0.72);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # --- SIDEBAR: Model Configuration ---
    with st.sidebar:
        st.title("⚙️ Configuration")

        st.subheader("Detection Model")
        det_family = BASE_CFG["models"]["detection"].get("name", DET_FAMILIES[0])
        det_variant_options = DET_VARIANTS_BY_FAMILY.get(det_family, ("n", "s", "m", "l", "x"))
        base_det_variant = BASE_CFG["models"]["detection"].get("variant", det_variant_options[0])
        det_variant_index = det_variant_options.index(base_det_variant) if base_det_variant in det_variant_options else 0
        det_variant = st.selectbox("Detection Variant", det_variant_options, index=det_variant_index)

        st.divider()
        st.subheader("Pose Estimation Model")
        base_pose_family = BASE_CFG["models"]["pose"].get("name", POSE_FAMILIES[0])
        pose_family_index = POSE_FAMILIES.index(base_pose_family) if base_pose_family in POSE_FAMILIES else 0
        pose_family = st.selectbox("Pose Family", POSE_FAMILIES, index=pose_family_index)
        pose_variant_options = POSE_VARIANTS.get(pose_family, ())
        base_pose_variant = BASE_CFG["models"]["pose"].get("variant", pose_variant_options[0] if pose_variant_options else "")
        pose_variant_index = pose_variant_options.index(base_pose_variant) if base_pose_variant in pose_variant_options else 0
        pose_variant = st.selectbox("Pose Variant", pose_variant_options, index=pose_variant_index)

        device = BASE_CFG["models"]["pose"].get("device", "cuda") or "cuda"
        save_video = False

        st.divider()
        st.subheader("SPARTA / Anomaly Model")
        branch_options = ["SPARTA-C", "SPARTA-F", "SPARTA-H"]
        default_branch_display = branch_to_display(DEFAULT_CFG.get("branch", SPARTA_CFG.get("branch", "SPARTA_C")))
        branch_index = branch_options.index(default_branch_display) if default_branch_display in branch_options else 0
        sparta_branch_display = st.selectbox("SPARTA Branch", branch_options, index=branch_index)
        sparta_branch_map = {
            "SPARTA-C": "SPARTA_C",
            "SPARTA-F": "SPARTA_F",
            "SPARTA-H": "SPARTA_H",
        }
        sparta_branch = sparta_branch_map.get(sparta_branch_display, sparta_branch_display)

        weightset_options = tuple(SPARTA_WEIGHTSETS.keys())
        default_weightset = infer_weightset_from_default()
        weightset_index = weightset_options.index(default_weightset) if default_weightset in weightset_options else 0
        selected_sparta_weightset = st.selectbox("Pretrained Weights", weightset_options, index=weightset_index)

        ckpt_c_default, ckpt_f_default, th_c_default, th_f_default = resolve_sparta_defaults(selected_sparta_weightset)
        default_th_c = cfg_float(DEFAULT_CFG, "eer_threshold_c", float(th_c_default if th_c_default is not None else 0.03))
        default_th_f = cfg_float(DEFAULT_CFG, "eer_threshold_f", float(th_f_default if th_f_default is not None else 0.06))
        default_th_h = cfg_float(DEFAULT_CFG, "eer_threshold_h", max(v for v in [default_th_c, default_th_f] if v is not None))

        threshold_c_input = st.slider("Threshold C", min_value=0.0, max_value=4.0, value=float(default_th_c or 0.0), step=0.01)
        threshold_f_input = st.slider("Threshold F", min_value=0.0, max_value=4.0, value=float(default_th_f or 0.0), step=0.01)
        threshold_h_input = st.slider("Threshold H", min_value=0.0, max_value=4.0, value=float(default_th_h or 0.0), step=0.01)

        if sparta_branch == "SPARTA_H":
            ckpt_c = ckpt_c_default
            ckpt_f = ckpt_f_default
            th_c = float(threshold_c_input)
            th_f = float(threshold_f_input)
            th_h = float(threshold_h_input)
            active_threshold = th_h
        elif sparta_branch == "SPARTA_F":
            ckpt_c = ""
            ckpt_f = ckpt_f_default
            th_c = None
            th_f = float(threshold_f_input)
            th_h = None
            active_threshold = th_f
        else:
            ckpt_c = ckpt_c_default
            ckpt_f = ""
            th_f = None
            th_c = float(threshold_c_input)
            th_h = None
            active_threshold = th_c
        st.caption(f"Active threshold: {active_threshold:.3f}")
    # --- MAIN: Title and Upload ---
    st.title("🎥 Human-Centric Surveillance System")
    st.caption("Real-time pose estimation with anomaly detection")
    
    # Video Source Section
    local_camera_available = camera_available(0)
    source_options = ["Upload Video"] + (["Use Camera"] if local_camera_available else [])
    source_mode = st.radio("Choose Input Source", source_options, horizontal=True)
    if not local_camera_available:
        st.caption("No local camera detected on this device. Upload mode is available.")

    upload = None
    video_path = None
    video_source_label = None
    source_stem = None

    if source_mode == "Upload Video":
        upload = st.file_uploader("📹 Upload Video", type=["mp4", "avi", "mov"], label_visibility="collapsed")
        if upload:
            video_path = save_upload(upload)
            video_source_label = str(video_path)
            source_stem = video_path.stem
    else:
        video_path = "__camera__:0"
        video_source_label = "Local Camera (0)"
        source_stem = "camera0"
        st.info("Live camera input selected. The app will use local camera index 0.")

    if video_path:
        # Start Analysis Button
        if st.button("🚀 Start Live Analysis", use_container_width=True, type="primary"):
            # 1. Setup Config
            run_cfg = build_run_config(BASE_CFG, pose_family, pose_variant, det_family, det_variant, device, video_path, save_video, sparta_branch, ckpt_c, ckpt_f, th_c, th_f, th_h)
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=POSE_DIR, delete=False) as tmp:
                yaml.safe_dump(run_cfg, tmp)
                tmp_path = tmp.name

            try:
                st.divider()
                
                # --- VIDEO DISPLAY AREA ---
                col_original, col_processed = st.columns(2, gap="small")
                
                with col_original:
                    st.markdown("#### Original video ")
                    if source_mode == "Upload Video" and upload is not None:
                        st.video(upload, start_time=0)
                    else:
                        st.info("Using live local camera stream.")
                
                with col_processed:
                    st.markdown("####  Model processing...")
                    frame_placeholder = st.empty()
                
                # --- PROGRESS AREA (Compact) ---
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                pipeline = PosePipeline(tmp_path)
                final_json_path = None
                
                try:
                    frame_count = 0
                    for frame, frame_id, total in pipeline.run_live():
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB")
                        
                        frame_count += 1
                        if total > 0 and frame_count % 50 == 0:
                            progress_bar.progress(min((frame_id + 1) / total, 1.0))
                    
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ Processing Complete: {frame_count} frames processed")
                    
                    # Get JSON path
                    res_prefix = run_cfg["paths"]["static_prefix"]
                    expected_json_name = get_expected_name(source_stem or "camera0", res_prefix) + run_cfg["paths"]["pose_json_suffix"]
                    final_json_path = Path(run_cfg["paths"]["pose_output_dir"]) / expected_json_name
                
                except Exception as e:
                    st.error(f"❌ Processing Error: {str(e)}")
                    status_text.error(f"Details: {traceback.format_exc()}")

                # --- ANOMALY SCORING ---
                st.divider()
                st.subheader("⚡ Anomaly Scores")
                
                if final_json_path and final_json_path.exists():
                    sparta_cfg = build_sparta_config(
                        BASE_CFG,
                        sparta_branch,
                        ckpt_c,
                        ckpt_f,
                        th_c,
                        th_f,
                        th_h,
                        final_json_path.parent,
                        device,
                        selected_sparta_weightset,
                    )
                    
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=BASE_DIR, delete=False) as tmp_sparta:
                        yaml.safe_dump(sparta_cfg, tmp_sparta)
                        tmp_sparta_path = tmp_sparta.name

                    with st.spinner("Computing anomaly scores..."):
                        subprocess.run(["python", "main.py", "--config", tmp_sparta_path], cwd=BASE_DIR, check=True)

                    scores_dir = Path(sparta_cfg["save_results_dir"])
                    scores_files = sorted(scores_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                    
                    if scores_files:
                        scores_path = scores_files[0]
                        df = pd.read_csv(scores_path)
                        threshold_line = active_threshold
                        st.pyplot(build_person_score_figure(df, threshold=threshold_line), use_container_width=True)
                        st.success(f"✅ Scores: {scores_path.name}")

                        with open(scores_path, "rb") as handle:
                            st.download_button(
                                "⬇️ Download Scores CSV",
                                data=handle.read(),
                                file_name=scores_path.name,
                                mime="text/csv",
                                use_container_width=True,
                            )

                else:
                    st.warning("⚠️ Pose JSON not found; skipping anomaly scoring.")
                    
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                import traceback
                st.error(f"Details: {traceback.format_exc()}")
            finally:
                if 'tmp_path' in locals(): 
                    Path(tmp_path).unlink(missing_ok=True)
                if 'tmp_sparta_path' in locals(): Path(tmp_sparta_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
