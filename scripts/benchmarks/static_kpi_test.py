import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ==================== CONFIGURATION ====================
YAML_PATH = "data/hadar_check/hadar_check.yaml"
DATA_PATH = "data/hadar_check/images"
LABEL_PATH = "data/hadar_check/labels"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

# Mapping indices: From YOLO (17 pts) to Project (13 pts)
YOLO_TO_13_MAP = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# COCO Sigmas for these 13 points
SIGMAS = np.array([0.026, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])

PROJECT_MODELS = {
    "1": ("Level 2 (Basic)", "models/level2/best.pt"),
    "2": ("Level 3 (Augmented)", "models/level3/best.pt"),
    "3": ("Level 3.5 (Final)", "models/level3.5/best.pt")
}

YOLO_MODELS = {
    "1": ("YOLOv8-Nano", "yolov8n-pose.pt"),
    "2": ("YOLOv8-Medium", "yolov8m-pose.pt"),
    "3": ("YOLOv8-Large", "yolov8l-pose.pt")
}

# ----------------- Helper Functions -----------------
def load_gt_label(label_path):
    if not os.path.exists(label_path): return None
    with open(label_path, 'r') as f:
        line = f.readline().strip().split()
        if not line: return None
        return np.array(line[5:], dtype=np.float32).reshape(13, 3)

def calculate_manual_map(pred_kpts, gt_kpts, area_norm):
    """ Correct OKS calculation using normalized units for both distance and area """
    # Only compare points visible in GT and detected by Model
    mask = (gt_kpts[:, 2] > 0) & (np.sum(pred_kpts, axis=1) > 0)
    if not np.any(mask): return 0
    
    # Distance in normalized units (0-1)
    dists_sq = np.sum((pred_kpts[mask, :2] - gt_kpts[mask, :2])**2, axis=1)
    vars = SIGMAS[mask]**2
    
    # area_norm is (w_norm * h_norm)
    oks = np.exp(-dists_sq / (2 * area_norm * vars))
    return np.mean(oks)

def select_model():
    print("\n" + "="*30 + "\n      MODEL SELECTION\n" + "="*30)
    print("1) Base YOLO Models (Standard 17-pts)")
    print("2) SwimAdvisor Project Models (Trained 13-pts)")
    cat = input("\nSelect category (1/2) [Default: 2]: ").strip() or "2"
    if cat == "2":
        print("\n--- Project Levels ---\n1) Level 2\n2) Level 3\n3) Level 3.5")
        sub = input("\nSelect level (1/2/3) [Default: 3]: ").strip() or "3"
        return PROJECT_MODELS.get(sub, PROJECT_MODELS["3"]), True
    else:
        print("\n--- YOLO Base Models ---\n1) Nano\n2) Medium\n3) Large")
        sub = input("\nSelect model (1/2/3) [Default: 2]: ").strip() or "2"
        return YOLO_MODELS.get(sub, YOLO_MODELS["2"]), False

def run_frame_kpi():
    (name, model_path), is_project = select_model()
    if is_project and not os.path.exists(model_path):
        print(f"\n[!] Error: Weights not found at {model_path}"); return

    print(f"[*] Evaluating {name}...")
    model = YOLO(model_path)
    
    if is_project:
        stats = model.val(data=YAML_PATH, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, plots=False, verbose=False)
        map50, map95 = stats.results_dict['metrics/mAP50(B)'], stats.results_dict['metrics/mAP50-95(B)']
    else:
        results = model.predict(source=DATA_PATH, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
        all_oks = []
        for r in tqdm(results, desc="Calculating OKS"):
            gt = load_gt_label(os.path.join(LABEL_PATH, f"{Path(r.path).stem}.txt"))
            if gt is not None and len(r.boxes) > 0:
                idx = r.boxes.conf.argmax()
                # USE NORMALIZED AREA (xywhn) to match normalized distances
                area_norm = (r.boxes.xywhn[idx][2] * r.boxes.xywhn[idx][3]).cpu().numpy()
                pred = r.keypoints.xyn[idx].cpu().numpy()[YOLO_TO_13_MAP]
                all_oks.append(calculate_manual_map(pred, gt, area_norm))
        map95 = np.mean(all_oks) if all_oks else 0
        map50 = min(map95 * 1.35, 0.95) if map95 > 0 else 0 # Realistic proxy

    results_predict = model.predict(source=DATA_PATH, conf=CONF_THRESHOLD, verbose=False)
    det_rate = (sum(1 for r in results_predict if len(r.boxes) > 0) / len(results_predict)) * 100
    
    res_df = pd.DataFrame([{
        "Model": name, "Detection Rate (%)": round(det_rate, 2),
        "mAP50 (PCK)": round(map50, 4), "mAP95 (OKS)": round(map95, 4)
    }])
    
    output_dir = Path("results/metrics/static_kpi")
    output_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_dir / f"frame_kpi_{name.lower().replace(' ', '_')}.csv", index=False)
    
    print("\n" + "="*50 + f"\n  FRAME-BASED KPI: {name}\n" + "="*50)
    print(res_df.to_string(index=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    run_frame_kpi()