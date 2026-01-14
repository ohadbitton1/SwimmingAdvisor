"""
Script to check and test YOLO models
Tests: yolo11n.pt and yolov8s-pose.pt
"""

from ultralytics import YOLO
import torch
import os
import time
from pathlib import Path

def get_model_info(model_path):
    """Get detailed information about a model"""
    print(f"\n{'='*60}")
    print(f"Checking: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: File not found: {model_path}")
        return None
    
    # Get file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📦 File size: {file_size:.2f} MB")
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Get model info
        print(f"✅ Model loaded successfully")
        print(f"📋 Model type: {type(model.model).__name__}")
        
        # Check if it's a pose model
        is_pose = hasattr(model.model, 'kpt_shape') or 'pose' in model_path.lower()
        print(f"🎯 Model task: {'Pose Estimation' if is_pose else 'Object Detection'}")
        
        # Get number of classes
        if hasattr(model.model, 'nc'):
            print(f"📊 Number of classes: {model.model.nc}")
        
        # Get class names
        if hasattr(model.model, 'names'):
            print(f"🏷️  Classes: {list(model.model.names.values())}")
        
        # Check input size
        if hasattr(model.model, 'args'):
            if hasattr(model.model.args, 'imgsz'):
                print(f"🖼️  Input image size: {model.model.args.imgsz}")
        
        # Check device capability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"💻 Device: {device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        return model
        
    except Exception as e:
        print(f"❌ ERROR loading model: {str(e)}")
        return None

def test_inference(model, model_name, test_image_path=None):
    """Test model inference on a sample image"""
    print(f"\n{'='*60}")
    print(f"Testing inference: {model_name}")
    print(f"{'='*60}")
    
    # Find a test image
    if test_image_path is None:
        # Try to find an image in the dataset
        possible_paths = [
            'Dataset/train/images',
            'Dataset/valid/images',
            'Dataset/test/images'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                images = list(Path(path).glob('*.jpg')) + list(Path(path).glob('*.png'))
                if images:
                    test_image_path = str(images[0])
                    break
    
    if test_image_path is None or not os.path.exists(test_image_path):
        print("⚠️  No test image found. Skipping inference test.")
        return None
    
    print(f"🖼️  Test image: {test_image_path}")
    
    try:
        # Warm up
        _ = model(test_image_path, verbose=False)
        
        # Time inference
        start_time = time.time()
        results = model(test_image_path, verbose=False)
        inference_time = time.time() - start_time
        
        print(f"⏱️  Inference time: {inference_time*1000:.2f} ms")
        
        # Get results info
        if results and len(results) > 0:
            result = results[0]
            print(f"📈 Detections found: {len(result.boxes) if hasattr(result, 'boxes') else 'N/A'}")
            
            # Check for keypoints if pose model
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                print(f"🦴 Keypoints detected: {len(result.keypoints)}")
        
        return results
        
    except Exception as e:
        print(f"❌ ERROR during inference: {str(e)}")
        return None

def compare_models():
    """Compare both models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    models_to_check = [
        ('yolo11n.pt', 'YOLO11 Nano'),
        ('yolov8s-pose.pt', 'YOLOv8 Small Pose')
    ]
    
    results = {}
    
    for model_path, model_name in models_to_check:
        model = get_model_info(model_path)
        if model:
            results[model_name] = {
                'model': model,
                'path': model_path
            }
            
            # Test inference
            test_results = test_inference(model, model_name)
            results[model_name]['inference_results'] = test_results
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model_name, data in results.items():
        print(f"\n✅ {model_name}:")
        print(f"   Path: {data['path']}")
        if 'inference_results' in data and data['inference_results']:
            print(f"   Status: Working ✓")
        else:
            print(f"   Status: Inference test skipped")
    
    return results

if __name__ == "__main__":
    print("🔍 YOLO Model Checker")
    print("="*60)
    
    # Check PyTorch and CUDA
    print(f"\n📦 Environment:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Compare models
    results = compare_models()
    
    print("\n" + "="*60)
    print("✅ Check complete!")
    print("="*60)
