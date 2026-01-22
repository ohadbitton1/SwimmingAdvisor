import numpy as np
import argparse
import os
from ultralytics import YOLO
from datetime import datetime

def calculate_jitter(kpts_series):
    """
    Calculates movement (Jitter) between consecutive frames.
    Matches original logic: np.linalg.norm(diffs, axis=2).
    """
    if len(kpts_series) < 2:
        return 0.0
    arr = np.array(kpts_series)  # Shape: (Frames, Keypoints, XY)
    diffs = np.diff(arr, axis=0)
    distances = np.linalg.norm(diffs, axis=2)
    return np.mean(distances)

def run_unified_evaluation(level, model_path, data_yaml, video_path):
    # Setup Output Directory
    report_dir = f"results/eval/level_{level}"
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\n" + "="*50)
    print(f"      SWIMADVISOR PERFORMANCE EVALUATION - LEVEL {level}")
    print(f"      Initializing system and loading model...")
    print("="*50)

    # Validate model path
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    model = YOLO(model_path)

    # ---------------------------------------------------------
    # PART 1: STATIC ACCURACY EVALUATION (OKS & PCK)
    # ---------------------------------------------------------
    print(f"\n[PHASE 1] Running Validation on Dataset: {data_yaml}")
    
    # Run validation using the 'test' split defined in the YAML file
    stats = model.val(
        data=data_yaml, 
        split='test', 
        plots=True, 
        project='results/eval', 
        name=f'level_{level}',
        exist_ok=True
    )
    
    oks = stats.pose.map     # mAP@50-95 (OKS)
    pck = stats.pose.map50   # mAP@50 (PCK)

    # =========================================================
    # PART 2: DYNAMIC STABILITY EVALUATION (JITTER)
    # =========================================================
    print(f"\n[PHASE 2] Running Temporal Jitter Analysis on Video...")
    print(f"Source: {video_path}")
    
    jitter = 0.0
    if os.path.exists(video_path):
        # Predict on video to extract keypoint sequences
        video_results = model.predict(source=video_path, stream=True, conf=0.25)
        kpts_history = []
        
        for res in video_results:
            if res.keypoints and len(res.keypoints.xy) > 0:
                # Target the primary detection (index 0)
                kpts_history.append(res.keypoints.xy[0].cpu().numpy())
        
        jitter = calculate_jitter(kpts_history)
    else:
        print(f"Warning: Test video not found at {video_path}. Jitter score set to 0.0")

    # ---------------------------------------------------------
    # REPORT GENERATION
    # ---------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file = os.path.join(report_dir, f"evaluation_report.txt")
    
    report_content = f"""
==================================================
      SWIMADVISOR PERFORMANCE REPORT - LEVEL {level}
==================================================
Timestamp:     {timestamp}
Model Path:    {model_path}
Dataset YAML:  {data_yaml}
Test Video:    {video_path}

--------------------------------------------------
METRICS SUMMARY
--------------------------------------------------
Static OKS (mAP 50-95): {oks:.4f}
Static PCK (mAP 50):    {pck:.4f}
Temporal Jitter:        {jitter:.2f} px/frame

--------------------------------------------------
STATUS
--------------------------------------------------
Evaluation completed. Plots and metrics are saved 
in: {report_dir}
==================================================
"""
    
    # Save the report to the results folder
    with open(report_file, "w") as f:
        f.write(report_content)

    print("\n" + "#"*50)
    print(f"      EVALUATION COMPLETE - LEVEL {level}")
    print("#"*50)
    print(f"Static OKS: {oks:.4f}")
    print(f"Static PCK: {pck:.4f}")
    print(f"Jitter:     {jitter:.2f} px/frame")
    print(f"Report saved to: {report_file}")
    print("#"*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluation Script for SwimAdvisor")

    # Defaults set for Level 3 as per current project status
    parser.add_argument("--level", type=int, default=3, help="Project level (1, 2, or 3)")
    parser.add_argument("--model", type=str, default="models/level3/best.pt", help="Path to best.pt")
    parser.add_argument("--data", type=str, default="data/augmented/data.yaml", help="Path to data.yaml")
    parser.add_argument("--video", type=str, default="data/swimming_test.mp4", help="Path to test video")

    args = parser.parse_args()

    run_unified_evaluation(args.level, args.model, args.data, args.video)