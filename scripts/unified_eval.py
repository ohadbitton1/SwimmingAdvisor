import numpy as np
from ultralytics import YOLO
import os

def calculate_jitter(kpts_series):
    """חישוב רעידות (Jitter) בין פריימים"""
    if len(kpts_series) < 2:
        return 0.0
    arr = np.array(kpts_series) # (Frames, Keypoints, XY)
    diffs = np.diff(arr, axis=0)
    distances = np.linalg.norm(diffs, axis=2)
    return np.mean(distances)

def run_evaluation():
    # --- הגדרת נתיבים לפי המבנה שלך ---
    model_path = 'yolov8n-pose.pt'
    data_yaml = 'data/freestyle/data.yaml'
    video_path = 'data/swimming_test.mp4'
    
    model = YOLO(model_path)
    
    # 1. מדדי דיוק סטטיים (OKS & PCK)
    print("\n--- Phase 1: OKS & PCK ---")
    stats = model.val(data=data_yaml, split='test', plots=False)
    oks = stats.box.map    # mAP@50-95 (OKS)
    pck = stats.box.map50  # mAP@50 (PCK)
    
    # 2. מדד יציבות דינמי (Jitter)
    print("\n--- Phase 2: Temporal Jitter ---")
    jitter = 0.0
    if os.path.exists(video_path):
        video_results = model.predict(source=video_path, stream=True, conf=0.25)
        kpts_history = []
        for res in video_results:
            if res.keypoints and len(res.keypoints.xy) > 0:
                # לוקחים את האדם הראשון שזוהה בפריים
                kpts_history.append(res.keypoints.xy[0].cpu().numpy())
        jitter = calculate_jitter(kpts_history)
    else:
        print(f"Error: Video not found at {video_path}")

    # --- דו"ח סופי ---
    print("\n" + "="*45)
    print(f"       LEVEL 1 - ZERO SHOT REPORT")
    print("="*45)
    print(f"Static OKS (mAP 50-95): {oks:.4f}")
    print(f"Static PCK (mAP 50):    {pck:.4f}")
    print(f"Temporal Jitter:        {jitter:.2f} px/frame")
    print("="*45)

if __name__ == "__main__":
    run_evaluation()