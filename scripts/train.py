from ultralytics import YOLO
import os

def run_training():
    # 1. Load the model - Using Nano version for efficient Fine-tuning
    model = YOLO('yolov8n-pose.pt')

    # 2. Define the path to the augmented data configuration
    data_path = 'data/augmented/data.yaml'
    #level2: data_path = 'data/freestyle/data.yaml'

    # 3. Start the training process for Level 3
    model.train(
        data=data_path,
        epochs=100,              # 100 epochs to ensure convergence with augmented data
        imgsz=640,               # Standard YOLO resolution
        batch=16,                # Optimal for T4 GPU (15GB VRAM)
        device=0,                # Use CUDA device 0
        project='runs/train',    # Root directory for results
        #name='level2_basic_FT'
        name='level3_augmented_FT', # Experiment name for Level 3
        save=True,               # Save weights (best.pt)
        plots=True,              # Generate training charts
        workers=4,               # Use multi-processing for data loading
        patience=20,             # Early stopping if no improvement for 20 epochs
        optimizer='SGD',         # Stable optimizer for pose estimation
        lr0=0.01                 # Initial learning rate
    )

if __name__ == "__main__":
    run_training()


    