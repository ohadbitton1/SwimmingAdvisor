from ultralytics import YOLO
import os

# טעינת המודל (יורד אוטומטית אם חסר)
model = YOLO('yolov8n-pose.pt')

# נתיב לתמונות המבחן לפי המבנה שלך
test_images_path = 'data/freestyle/test/images'

print(f"Running visual inference on: {test_images_path}")
model.predict(
    source=test_images_path,
    save=True,
    conf=0.25,
    name='level1_visual_results' # יישמר ב-runs/pose/level1_visual_results
)

print("Done! Check 'runs/pose/level1_visual_results' for skeleton images.")