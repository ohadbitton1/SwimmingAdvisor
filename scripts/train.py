from ultralytics import YOLO
import os

def run_training():
    # 1. Load the model 
    # We use 'yolov8m-pose.pt' (Medium) because it offers much better keypoint accuracy 
    # than 'n' (Nano) for limbs and joints, while still being fast enough for video analysis.
    print("Loading YOLOv8m-pose model...")
    model = YOLO('yolov8m-pose.pt')

    # 2. Define the path to the NEW unified configuration file
    # This points to the clean dataset we just created (single class, 13 keypoints).
    data_path = '/content/datasets/unified_13pt.yaml'

    # 3. Start the training process
    print(f"Starting training on {data_path}...")
    
    model.train(
        data=data_path,
        epochs=50,               # 50 epochs is a good start. Increase to 100 if results are good.
        imgsz=640,               # Standard YOLO input resolution.
        batch=16,                # Optimal batch size for a standard Colab T4 GPU.
        device=0,                # Ensure we are using the GPU.
        
        # === Output Configuration (Crucial for Colab) ===
        # Save results directly to Google Drive so they aren't lost if Colab disconnects.
        project='/content/drive/MyDrive/SwimAdvisor/models', 
        name='swim_pose_final_v1', 
        
        # === Training Settings ===
        save=True,               # Save the best model checkpoints (best.pt).
        plots=True,              # Generate loss and accuracy graphs.
        workers=2,               # Using 2 workers is more stable in Colab than 4 or 8.
        patience=15,             # Early Stopping: Stop if no improvement for 15 epochs.
        optimizer='auto',        # YOLO automatically selects the best optimizer (usually AdamW).
        
        # === Swimming-Specific Augmentations ===
        degrees=15.0,            # Random rotation (+/- 15 deg) to handle non-horizontal swimming angles.
        flipud=0.0,              # Disable Up-Down flip (swimmers are rarely upside down relative to camera).
        fliplr=0.5,              # Enable Left-Right flip (50% chance) to learn symmetry.
        mosaic=1.0,              # Enable Mosaic to help the model learn from complex scenes.
    )

    print("\n[SUCCESS] Training finished! Check your Google Drive 'SwimAdvisor/models' folder for 'best.pt'.")

if __name__ == "__main__":
    run_training()