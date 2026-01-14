import numpy as np
from ultralytics import YOLO
import os
##SAME TEST- JUST DIFFERENT PATH AND PRINTS
def calculate_jitter(kpts_series):
    """חישוב רעידות (Jitter) בין פריימים"""
    if len(kpts_series) < 2:
        return 0.0
    arr = np.array(kpts_series) # (Frames, Keypoints, XY)
    diffs = np.diff(arr, axis=0)
    distances = np.linalg.norm(diffs, axis=2)
    return np.mean(distances)

def run_evaluation():
    # --- הגדרת נתיבים מעודכנים לרמה 2 ---
    # וודא שהקובץ נמצא בתיקיית models/level2/ בתוך הפרויקט שלך
    model_path = 'models/level2/best.pt' 
    data_yaml = 'data/freestyle/data.yaml'
    video_path = 'data/swimming_test.mp4'
    
    # טעינת המודל המאומן
    if not os.path.exists(model_path):
        print(f"Error: Fine-tuned model not found at {model_path}")
        return

    model = YOLO(model_path)
    
    # 1. מדדי דיוק סטטיים (OKS & PCK)
    print("\n--- Phase 1: OKS & PCK (Evaluation on Test Set) ---")
    # הרצה על ה-Test split כפי שהוגדר ב-YAML
    stats = model.val(data=data_yaml, split='test', plots=False)
    
    # ב-YOLOv8 Pose, המדדים נמצאים תחת stats.pose
    oks = stats.pose.map     # mAP@50-95 (Keypoints)
    pck = stats.pose.map50   # mAP@50 (Keypoints)
    
    # 2. מדד יציבות דינמי (Jitter)
    print("\n--- Phase 2: Temporal Jitter (Evaluation on Video) ---")
    jitter = 0.0
    if os.path.exists(video_path):
        video_results = model.predict(source=video_path, stream=True, conf=0.25)
        kpts_history = []
        for res in video_results:
            if res.keypoints and len(res.keypoints.xy) > 0:
                # לוקחים את האדם הראשון שזוהה בפריים (השחיין המרכזי)
                # המרה ל-numpy לצורך חישוב מרחקים
                kpts_history.append(res.keypoints.xy[0].cpu().numpy())
        
        jitter = calculate_jitter(kpts_history)
    else:
        print(f"Error: Video not found at {video_path}")

    # --- דו"ח סופי משופר לרמה 2 ---
    print("\n" + "="*45)
    print(f"      LEVEL 2 - FINE-TUNED MODEL REPORT")
    print(f"      Model Source: {model_path}")
    print("="*45)
    print(f"Static OKS (mAP 50-95): {oks:.4f}")
    print(f"Static PCK (mAP 50):    {pck:.4f}")
    print(f"Temporal Jitter:        {jitter:.2f} px/frame")
    print("="*45)
    print(f"Comparison to Level 1 Baseline (Target: >0.8488 PCK)")
    print("="*45)

if __name__ == "__main__":
    run_evaluation()