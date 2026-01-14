from ultralytics import YOLO
import torch

def train_swimmer_model():
    # 1. Load the architecture with pre-trained weights
    # 'yolov8s-pose.pt' is the 'Small' version—good balance of speed/accuracy
    model = YOLO('yolov8m-pose.pt') 

    # 2. Connect and Train
    model.train(
        data='Dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='swimmer_project',
        name='v1_baseline',
        save=True
    )

if __name__ == "__main__":
    import sys
    
    # Check for colab flag
    colab_mode = '--colab' in sys.argv or is_colab()
    
    train_swimmer_model(colab=colab_mode)