import os
import time
import numpy as np
from ultralytics import YOLO

def run_final_level1_report():
    
    model_path = 'yolov8n-pose'
    test_images_path = 'data/freestyle/test/images'
    
    
    model = YOLO(model_path)
    
    
    image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    
    print(f"Starting Level 1 Final Report on {total_images} images...")
    
    
    detections_count = 0
    all_confidences = []
    inference_times = []
    
    
    results = model.predict(source=test_images_path, stream=True, conf=0.1) # סף נמוך כדי לראות הכל
    
    for res in results:
        
        inference_times.append(res.speed['inference'])
        
        
        if len(res.boxes) > 0:
            detections_count += 1
            
            top_conf = res.boxes.conf[0].item()
            all_confidences.append(top_conf)

    
    detection_rate = (detections_count / total_images) * 100 if total_images > 0 else 0
    avg_conf = np.mean(all_confidences) if all_confidences else 0
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    
    print("\n" + "="*45)
    print("       LEVEL 1 - FINAL SUMMARY REPORT")
    print("="*45)
    print(f"Total Test Images:     {total_images}")
    print(f"Images with Detection: {detections_count}")
    print(f"Detection Rate:        {detection_rate:.2f}%")
    print(f"Average Confidence:    {avg_conf:.4f}")
    print(f"Avg Inference Speed:   {avg_inference_time:.2f} ms/image")
    print("-" * 45)
    print("Metrical Note: OKS/PCK not available due to")
    print("topology mismatch (17 vs 13 keypoints).")
    print("="*45)

if __name__ == "__main__":
    run_final_level1_report()