import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
from tqdm import tqdm

# Import our modular logic
from scripts.smoothing_layer import (
    global_antiflip, torso_alignment_guard, leg_chain_guard,
    safe_midpoint, dist, enforce_min_width_by_torso,
    apply_temporal_smoothing, any_nan,
    SHOULDER_WIDTH_RATIO, HIP_WIDTH_RATIO, KNEE_WIDTH_RATIO, ANKLE_WIDTH_RATIO
)

# ==================== PATH CONFIG ====================
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/SwimAdvisor"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_VIDEO  = os.path.join(BASE_DIR, "data", "hadar_video.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "results", "vids", "swimadvisor_final.mp4")
MODEL_PATH   = os.path.join(BASE_DIR, "models", "level3.5", "best.pt")

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

CONF_BODY = 0.25
CONF_HANDS = 0.01

def draw_fancy_skeleton(frame, kpts):
    overlay = frame.copy()
    skeleton = [(0,1), (0,2), (1,2), (1,3), (3,5), (2,4), (4,6), (1,7), (2,8), (7,8), (7,9), (9,11), (8,10), (10,12)]
    for p1, p2 in skeleton:
        if not any_nan(kpts[p1], kpts[p2]):
            color = (255, 120, 0) if p2 % 2 == 0 else (0, 180, 255)
            cv2.line(overlay, tuple(kpts[p1].astype(int)), tuple(kpts[p2].astype(int)), color, 2, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

def process_swim_v10():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}"); return
    
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_VIDEO}"); return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    raw_data = []
    print("Step 1: AI Inference...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = model.predict(frame, imgsz=640, conf=CONF_HANDS, verbose=False)[0]
        f_kpts = np.full((13, 2), np.nan)
        if res.boxes:
            b_idx = res.boxes.conf.argmax().item()
            d = res.keypoints.data[b_idx].cpu().numpy()
            for i in range(13):
                if d[i, 2] >= (CONF_HANDS if i in [5,6] else CONF_BODY): f_kpts[i] = d[i, :2]
        raw_data.append(f_kpts)

    print("Step 2: V10 Stabilization & Guards...")
    proc_data = np.array(raw_data)
    prev = None
    for f in range(len(proc_data)):
        curr = proc_data[f].copy()
        
        # Applying Guards from smoothing_layer
        curr = global_antiflip(curr, prev)
        curr = torso_alignment_guard(curr)
        curr = leg_chain_guard(curr)
        
        # Anatomical constraints
        neck, pelv = safe_midpoint(curr[1], curr[2]), safe_midpoint(curr[7], curr[8])
        if not any_nan(neck, pelv):
            t_len = dist(neck, pelv)
            if t_len > 20:
                curr = enforce_min_width_by_torso(curr, 1, 2, t_len * SHOULDER_WIDTH_RATIO)
                curr = enforce_min_width_by_torso(curr, 7, 8, t_len * HIP_WIDTH_RATIO)
                curr = enforce_min_width_by_torso(curr, 9, 10, t_len * KNEE_WIDTH_RATIO)
                curr = enforce_min_width_by_torso(curr, 11, 12, t_len * ANKLE_WIDTH_RATIO)
        
        proc_data[f], prev = curr, curr

    print("Step 3: Temporal Smoothing...")
    proc_data = apply_temporal_smoothing(proc_data)

    print(f"Step 4: Rendering to {OUTPUT_VIDEO}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(len(proc_data)):
        ret, frame = cap.read()
        if not ret: break
        out.write(draw_fancy_skeleton(frame, proc_data[f]))
    
    cap.release(); out.release()
    print("✅ Done!")

if __name__ == "__main__":
    process_swim_v10()