import cv2
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
VIDEO_PATH = r"/home/waleed64/Documents/Human_Centric_Anomaly_Detection_Agent/videos/surveillance video.mp4"
JSON_PATH = "fighting_vit_poses.json"
RESULTS_CSV = "anomaly_fighting_results.csv"
OUTPUT_VIDEO = "fighting_sparta_visualized.avi"

# COCO 17 Connections
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), # Legs
    (5, 11), (6, 12), (5, 6),                      # Torso
    (5, 7), (6, 8), (7, 9), (8, 10),               # Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),          # Head
    (3, 5), (4, 6)                                 # Ears to Shoulders
]

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def visualize():
    print("Loading data...")
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} not found. Run sparta_adapter_vit.py first.")
        return

    # Load anomaly scores
    results_df = pd.read_csv(RESULTS_CSV)
    
    # Load pose data
    with open(JSON_PATH, 'r') as f:
        pose_data = json.load(f)
    pose_df = pd.DataFrame(pose_data)

    # Load Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames into {OUTPUT_VIDEO}...")

    # Dictionary for fast lookup: {(frame_id, person_id): score}
    score_lookup = {(row.frame_id, row.person_id): row.anomaly_score for row in results_df.itertuples()}

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get all poses for this frame
        current_poses = pose_df[pose_df['frame_id'] == frame_id]

        for _, pose in current_poses.iterrows():
            person_id = pose['person_id']
            score = score_lookup.get((frame_id, person_id), 0.0)

            # Color mapping: Green to Red based on score
            # SHT models usually have a base loss, thresholding around 0.1-0.2 for clear anomalies
            # Let's use a normalized color scale
            color_val = min(int(score * 2000), 255) # Scale for visualization
            color = (0, 255 - color_val, color_val) # Green -> Red

            # Extract keypoints
            kpts = []
            for name in COCO_KEYPOINT_NAMES:
                kpts.append((int(pose[f"{name}_x"]), int(pose[f"{name}_y"])))

            # Draw Skeleton
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                pt1 = kpts[start_idx]
                pt2 = kpts[end_idx]
                if pt1 != (0, 0) and pt2 != (0, 0): # Basic check for valid keypoints
                    cv2.line(frame, pt1, pt2, color, 2)

            # Draw Joints
            for pt in kpts:
                if pt != (0, 0):
                    cv2.circle(frame, pt, 3, color, -1)

            # Draw Label
            cv2.putText(frame, f"ID:{person_id} Score:{score:.4f}", (kpts[5][0], kpts[5][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"Progress: {frame_id}/{total_frames}")

    cap.release()
    out.release()
    print(f"✅ DONE! Visualized video saved at: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    visualize()
