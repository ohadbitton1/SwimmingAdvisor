import os
import shutil
from roboflow import Roboflow

# Define the target directory path
target_path = "data/augmented"

# Download the augmented dataset (Version 2) from Roboflow
rf = Roboflow(api_key="YOUR API KEY")
project = rf.workspace("swimadvisor").project("freestyle-kkgfa-ouusu")
version = project.version(2)
dataset = version.download("yolov8")

# Organize files into data/augmented
if os.path.exists(target_path):
    shutil.rmtree(target_path)

os.makedirs("data", exist_ok=True)
shutil.move(dataset.location, target_path)

print(f"Dataset is ready at: {os.path.abspath(target_path)}")