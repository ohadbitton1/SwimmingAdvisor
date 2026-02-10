import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import cv2
from pathlib import Path

# ==================== SETTINGS ====================
VIDEO_PATH = "data/hadar_video.mp4"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25          
KPT_CONF_THRESHOLD = 0.30      
IOU_TRACK_THRESHOLD = 0.30     
MAX_MISSES = 10                

# Mapping: Indices from YOLO (17 pts) to match Project (13 pts)
YOLO_TO_13_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
KPT_NAMES_13 = [
    "Head", "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow", 
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip", 
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle"
]

PROJECT_MODELS = {
    "1": ("Level 2", "models/level2/best.pt"),
    "2": ("Level 3", "models/level3/best.pt"),
    "3": ("Level 3.5 (Final)", "models/level3.5/best.pt")
}

YOLO_MODELS = {
    "1": ("YOLOv8-Nano", "yolov8n-pose.pt"),
    "2": ("YOLOv8-Medium", "yolov8m-pose.pt"),
    "3": ("YOLOv8-Large", "yolov8l-pose.pt")
}

# ==================== HELPER FUNCTIONS ====================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def normalize_kpts(kpts_xy, box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    out = kpts_xy.copy()
    out[..., 0] = (out[..., 0] - x1) / w
    out[..., 1] = (out[..., 1] - y1) / h
    return out

def calculate_jitter_accel(kpts_norm, fps):
    dt = 1.0 / max(1e-6, fps)
    v = np.diff(kpts_norm, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    mag = np.sqrt(a[..., 0]**2 + a[..., 1]**2)
    return np.nanmean(mag, axis=0)

def select_model():
    print("\n" + "="*30 + "\n      JITTER ANALYSIS\n" + "="*30)
    print("1) Base YOLO Models\n2) Project Models")
    cat = input("\nSelect category (1/2) [Default: 2]: ").strip() or "2"
    
    if cat == "2":
        print("\n--- Project Levels ---\n1) Level 2\n2) Level 3\n3) Level 3.5")
        sub = input("\nSelect level (1/2/3) [Default: 3]: ").strip() or "3"
        return PROJECT_MODELS.get(sub, PROJECT_MODELS["3"]), True
    else:
        print("\n--- YOLO Models ---\n1) Nano\n2) Medium\n3) Large")
        sub = input("\nSelect model (1/2/3) [Default: 2]: ").strip() or "2"
        return YOLO_MODELS.get(sub, YOLO_MODELS["2"]), False

# ==================== MAIN ANALYSIS ====================
def run_jitter_benchmark():
    (name, model_path), is_project = select_model()
    
    if is_project and not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}"); return

    model = YOLO(model_path)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at {VIDEO_PATH}"); return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    track_box, misses, kpts_series = None, 0, []
    num_kpts = 13 if is_project else 17

    print(f"[*] Analyzing stability for: {name}")
    for _ in tqdm(range(total_frames), desc="Inference"):
        ret, frame = cap.read()
        if not ret: break

        res = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        frame_kpts = np.full((num_kpts, 2), np.nan)

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            
            if track_box is None: 
                idx = int(np.argmax(confs))
            else:
                ious = np.array([iou_xyxy(track_box, b) for b in boxes])
                idx = int(np.argmax(ious)) if np.max(ious) >= IOU_TRACK_THRESHOLD else None

            if idx is not None:
                track_box, misses = boxes[idx], 0
                kpt_data = res.keypoints.data[idx].cpu().numpy()
                kpt_xy, kpt_c = kpt_data[:, :2], kpt_data[:, 2]
                good = kpt_c >= KPT_CONF_THRESHOLD
                kpt_xy[~good] = np.nan
                frame_kpts = normalize_kpts(kpt_xy, track_box)
            else:
                misses += 1
                if misses > MAX_MISSES: track_box = None

        kpts_series.append(frame_kpts)

    cap.release()
    kpts_np = np.stack(kpts_series)
    
    # Calculate Jitter and map to 13 points if it's a 17-pt YOLO model
    jitter = calculate_jitter_accel(kpts_np, fps)
    
    if not is_project:
        jitter = jitter[YOLO_TO_13_INDICES]

    # Reporting
    report = pd.DataFrame({
        "Keypoint": KPT_NAMES_13,
        "Jitter_Score": jitter
    }).sort_values("Jitter_Score", ascending=False)

    out_dir = Path("results/metrics/jitter_res")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / f"jitter_{name.lower().replace(' ', '_')}.csv"
    report.to_csv(csv_file, index=False)

    print(f"\nFinal Stability Report: {name}")
    print(report.to_string(index=False))
    print(f"\nFull report saved to: {csv_file}")

if __name__ == "__main__":
    run_jitter_benchmark()