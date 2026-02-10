import sys
import os
# Adds the root directory (SwimAdvisor) to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Prevent Python from creating __pycache__ folders
sys.dont_write_bytecode = True
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm
import shutil

# Import the modular logic from your new smoothing layer
from scripts.smoothing_layer import (
    global_antiflip, torso_alignment_guard, leg_chain_guard,
    safe_midpoint, dist, enforce_min_width_by_torso,
    apply_temporal_smoothing,
    SHOULDER_WIDTH_RATIO, HIP_WIDTH_RATIO, KNEE_WIDTH_RATIO, ANKLE_WIDTH_RATIO
)

# ==================== CONFIGURATION ====================
VIDEO_PATH = "data/hadar_video.mp4"
MODEL_PATH = "models/level3.5/best.pt"
PLOT_DIR = "results/plots"
METRIC_DIR = "results/metrics"

CONF_BODY = 0.25
CONF_HANDS = 0.01

KPT_NAMES = [
    "Head", "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow", 
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip", 
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle"
]

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

def calculate_jitter(data_np):
    """ Calculates Jitter based on the magnitude of acceleration (2nd derivative) """
    if len(data_np) < 3: return np.zeros(13)
    # Velocity (1st derivative)
    v = np.diff(data_np, axis=0)
    # Acceleration (2nd derivative)
    a = np.diff(v, axis=0)
    # Magnitude of acceleration
    mag = np.sqrt(a[..., 0]**2 + a[..., 1]**2)
    # Average jitter per keypoint, ignoring NaNs
    return np.nanmean(mag, axis=0)

def run_smoothing_impact_analysis():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}"); return

    print(f"[*] Loading Model and Video for Impact Analysis...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    raw_history = []
    v10_history = []
    prev_v10 = None

    # --- Phase 1: Inference & Geometric Guards ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[1/3] Running Inference and V10 Guards...")
    
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        
        # Inference
        res = model.predict(frame, imgsz=640, conf=CONF_HANDS, verbose=False)[0]
        curr_raw = np.full((13, 2), np.nan)
        
        if res.boxes:
            idx = res.boxes.conf.argmax().item()
            d = res.keypoints.data[idx].cpu().numpy()
            for i in range(13):
                # Apply same confidence logic as main.py
                if d[i, 2] >= (CONF_HANDS if i in [5,6] else CONF_BODY):
                    curr_raw[i] = d[i, :2]
        
        raw_history.append(curr_raw.copy())
        
        # Apply Structural Guards (The "Intelligence" of your layer)
        proc = curr_raw.copy()
        proc = global_antiflip(proc, prev_v10)
        proc = torso_alignment_guard(proc)
        proc = leg_chain_guard(proc)
        
        # Anatomical Scaling
        neck, pelv = safe_midpoint(proc[1], proc[2]), safe_midpoint(proc[7], proc[8])
        t_len = dist(neck, pelv)
        if not np.isnan(t_len) and t_len > 20:
            proc = enforce_min_width_by_torso(proc, 1, 2, t_len * SHOULDER_WIDTH_RATIO)
            proc = enforce_min_width_by_torso(proc, 7, 8, t_len * HIP_WIDTH_RATIO)
            proc = enforce_min_width_by_torso(proc, 9, 10, t_len * KNEE_WIDTH_RATIO)
            proc = enforce_min_width_by_torso(proc, 11, 12, t_len * ANKLE_WIDTH_RATIO)
        
        v10_history.append(proc)
        prev_v10 = proc

    cap.release()

    # --- Phase 2: Temporal Smoothing ---
    print("[2/3] Applying Savitzky-Golay Temporal Smoothing...")
    raw_np = np.array(raw_history)
    # Apply interpolation and Savgol via the smoothing_layer function
    v10_np = apply_temporal_smoothing(np.array(v10_history))

    # --- Phase 3: Metrics & Visualization ---
    print("[3/3] Calculating Stability Metrics and Generating Plots...")
    
    raw_jitter = calculate_jitter(raw_np)
    v10_jitter = calculate_jitter(v10_np)

    # 1. Generate CSV Report
    results = []
    for i in range(13):
        improvement = ((raw_jitter[i] - v10_jitter[i]) / raw_jitter[i]) * 100 if raw_jitter[i] > 0 else 0
        results.append({
            "Keypoint": KPT_NAMES[i],
            "Raw_Jitter_px": round(raw_jitter[i], 2),
            "V10_Jitter_px": round(v10_jitter[i], 2),
            "Improvement_%": round(improvement, 2)
        })
    
    df = pd.DataFrame(results)
    report_path = os.path.join(METRIC_DIR, "smoothing_impact_report.csv")
    df.to_csv(report_path, index=False)

    # 2. Generate Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(KPT_NAMES))
    width = 0.35

    plt.bar(x - width/2, raw_jitter, width, label='Raw Model (Unstable)', color='salmon')
    plt.bar(x + width/2, v10_jitter, width, label='Model + V10 Layer (Stable)', color='skyblue')

    plt.xlabel('Keypoints')
    plt.ylabel('Average Jitter (Pixels per Frame²)')
    plt.title('Stability Analysis: Smoothing Layer Impact')
    plt.xticks(x, KPT_NAMES, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "jitter_comparison.png")
    plt.savefig(plot_path)
    
    # Summary Output
    avg_imp = df["Improvement_%"].mean()
    print("\n" + "="*50)
    print(f"        SMOOTHING IMPACT SUMMARY")
    print("="*50)
    print(f"Total Jitter Reduction: {avg_imp:.2f}%")
    print(f"Most Improved: {df.loc[df['Improvement_%'].idxmax(), 'Keypoint']} ({df['Improvement_%'].max():.1f}%)")
    print("-" * 50)
    print(f"Plot saved: {plot_path}")
    print(f"CSV saved:  {report_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_smoothing_impact_analysis()
    # Cleanup: Remove the 'runs' directory if it was created by YOLO
    runs_path = os.path.join(os.getcwd(), "runs")
    if os.path.exists(runs_path):
        shutil.rmtree(runs_path)
        print("[*] Temporary 'runs' folder cleaned up.")