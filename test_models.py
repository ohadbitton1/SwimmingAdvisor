"""
Test YOLO models on images/videos
Usage: python test_models.py
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path
import cv2

def test_model_on_image(model_path, image_path, output_dir="test_results"):
    """Test a model on a single image and save results"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model
        model = YOLO(model_path)
        model_name = Path(model_path).stem
        
        # Run inference
        print("🔄 Running inference...")
        results = model(image_path, save=True, project=output_dir, name=model_name)
        
        # Get output path
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, model_name, f"{image_name}.jpg")
        
        print(f"✅ Results saved to: {output_path}")
        
        # Print detection info
        if results and len(results) > 0:
            result = results[0]
            
            # Check for boxes (detections)
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_detections = len(result.boxes)
                print(f"📦 Detections: {num_detections}")
                
                # Show confidence scores
                if num_detections > 0:
                    confidences = result.boxes.conf.cpu().numpy()
                    print(f"   Confidence range: {confidences.min():.2f} - {confidences.max():.2f}")
            
            # Check for keypoints (pose)
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                num_keypoints = len(result.keypoints)
                print(f"🦴 Keypoints detected: {num_keypoints}")
                if num_keypoints > 0:
                    print(f"   Keypoints per person: {result.keypoints.data.shape[1] if len(result.keypoints.data.shape) > 1 else 'N/A'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_model_on_folder(model_path, folder_path, max_images=None, output_dir="test_results", show_progress=True):
    """Test model on multiple images from a folder"""
    print(f"\n{'='*60}")
    print(f"Testing on folder: {folder_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    for ext in image_extensions:
        images.extend(Path(folder_path).glob(f'*{ext}'))
        images.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    # Sort images by name
    images = sorted(images)
    
    if not images:
        print(f"❌ No images found in: {folder_path}")
        return
    
    print(f"📸 Found {len(images)} images")
    
    # Load model once
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    # Test on images
    if max_images is None:
        test_images = images
        print(f"🧪 Testing on ALL {len(test_images)} images...")
    else:
        test_images = images[:max_images]
        print(f"🧪 Testing on {len(test_images)} images...")
    
    # Run batch inference for speed
    print("🔄 Running inference...")
    image_paths = [str(img) for img in test_images]
    
    try:
        # Use batch processing for better performance
        results = model(image_paths, save=True, project=output_dir, name=model_name, verbose=False)
        
        print(f"✅ Processed {len(results)} images")
        print(f"📁 Results saved to: {output_dir}/{model_name}/")
        
        # Summary statistics
        total_detections = 0
        total_keypoints = 0
        
        for i, result in enumerate(results):
            if hasattr(result, 'boxes') and result.boxes is not None:
                total_detections += len(result.boxes)
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                total_keypoints += len(result.keypoints)
        
        print(f"\n📊 Summary:")
        print(f"   Total detections: {total_detections}")
        if total_keypoints > 0:
            print(f"   Total keypoints: {total_keypoints}")
        
    except Exception as e:
        # Fallback to individual processing if batch fails
        print(f"⚠️  Batch processing failed, processing individually...")
        print(f"   Error: {str(e)}")
        
        for i, img_path in enumerate(test_images, 1):
            if show_progress:
                print(f"\n[{i}/{len(test_images)}] Testing: {img_path.name}")
            test_model_on_image(model_path, str(img_path), output_dir)

def compare_models_on_image(image_path, output_dir="test_results"):
    """Compare both models on the same image"""
    print(f"\n{'='*60}")
    print("COMPARING BOTH MODELS")
    print(f"{'='*60}")
    
    models = [
        ('yolo11n.pt', 'YOLO11 Nano'),
        ('yolov8s-pose.pt', 'YOLOv8 Small Pose')
    ]
    
    results = {}
    
    for model_path, model_name in models:
        if os.path.exists(model_path):
            print(f"\n🔍 Testing {model_name}...")
            result = test_model_on_image(model_path, image_path, output_dir)
            results[model_name] = result
        else:
            print(f"⚠️  {model_path} not found, skipping...")
    
    print(f"\n{'='*60}")
    print("✅ Comparison complete!")
    print(f"📁 Check results in: {output_dir}/")
    print(f"{'='*60}")

def compare_models_on_folder(folder_path, max_images=None, output_dir="test_results"):
    """Compare both models on a folder of images"""
    print(f"\n{'='*60}")
    print("COMPARING BOTH MODELS ON FOLDER")
    print(f"{'='*60}")
    
    models = [
        ('yolo11n.pt', 'YOLO11 Nano'),
        ('yolov8s-pose.pt', 'YOLOv8 Small Pose')
    ]
    
    for model_path, model_name in models:
        if os.path.exists(model_path):
            print(f"\n{'='*60}")
            print(f"Testing {model_name}...")
            print(f"{'='*60}")
            test_model_on_folder(model_path, folder_path, max_images=max_images, output_dir=output_dir, show_progress=False)
        else:
            print(f"⚠️  {model_path} not found, skipping...")
    
    print(f"\n{'='*60}")
    print("✅ Comparison complete!")
    print(f"📁 Check results in: {output_dir}/")
    print(f"{'='*60}")

def main():
    """Main test function"""
    print("🧪 YOLO Model Tester")
    print("="*60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Find available folders
    available_folders = []
    test_folders = [
        'frames_mi0.1',  # Extracted frames
        'Dataset/test/images',
        'Dataset/valid/images',
        'Dataset/train/images'
    ]
    
    for folder in test_folders:
        if os.path.exists(folder):
            # Check if it has images
            images = list(Path(folder).glob('*.jpg')) + list(Path(folder).glob('*.png'))
            if images:
                available_folders.append(folder)
    
    if not available_folders:
        print("❌ No image folders found!")
        return
    
    # Show available folders
    print(f"\n📁 Available folders:")
    for i, folder in enumerate(available_folders, 1):
        image_count = len(list(Path(folder).glob('*.jpg')) + list(Path(folder).glob('*.png')))
        print(f"   {i}. {folder} ({image_count} images)")
    
    # Get folder choice
    try:
        folder_choice = input(f"\nSelect folder (1-{len(available_folders)}, default 1): ").strip()
        folder_idx = int(folder_choice) - 1 if folder_choice else 0
        if folder_idx < 0 or folder_idx >= len(available_folders):
            folder_idx = 0
        selected_folder = available_folders[folder_idx]
    except:
        selected_folder = available_folders[0]
    
    # Get first image from selected folder
    images = sorted(list(Path(selected_folder).glob('*.jpg')) + list(Path(selected_folder).glob('*.png')))
    if not images:
        print("❌ No images found!")
        return
    
    test_image = str(images[0])
    image_count = len(images)
    
    print(f"\n📸 Selected folder: {selected_folder}")
    print(f"📸 Total images: {image_count}")
    print(f"📸 Sample image: {Path(test_image).name}")
    
    # Menu
    print("\n" + "="*60)
    print("TEST OPTIONS:")
    print("="*60)
    print("1. Test yolo11n.pt on single image")
    print("2. Test yolov8s-pose.pt on single image")
    print("3. Compare both models on single image")
    print("4. Test yolo11n.pt on folder (first 10 images)")
    print("5. Test yolov8s-pose.pt on folder (first 10 images)")
    print("6. Test yolo11n.pt on ALL images in folder")
    print("7. Test yolov8s-pose.pt on ALL images in folder")
    print("8. Compare both models on folder (first 10 images)")
    print("9. Compare both models on ALL images in folder")
    print("="*60)
    
    choice = input("\nEnter choice (1-9) or press Enter for option 3 (compare single): ").strip()
    
    if choice == "1":
        test_model_on_image('yolo11n.pt', test_image)
    elif choice == "2":
        test_model_on_image('yolov8s-pose.pt', test_image)
    elif choice == "3" or choice == "":
        compare_models_on_image(test_image)
    elif choice == "4":
        test_model_on_folder('yolo11n.pt', selected_folder, max_images=10)
    elif choice == "5":
        test_model_on_folder('yolov8s-pose.pt', selected_folder, max_images=10)
    elif choice == "6":
        test_model_on_folder('yolo11n.pt', selected_folder, max_images=None)
    elif choice == "7":
        test_model_on_folder('yolov8s-pose.pt', selected_folder, max_images=None)
    elif choice == "8":
        compare_models_on_folder(selected_folder, max_images=10)
    elif choice == "9":
        confirm = input(f"⚠️  Process ALL {image_count} images? This may take a while. (y/n): ").strip().lower()
        if confirm == 'y':
            compare_models_on_folder(selected_folder, max_images=None)
        else:
            print("Cancelled.")
    else:
        print("Invalid choice. Running comparison on single image...")
        compare_models_on_image(test_image)
    
    print("\n✅ Testing complete!")
    print(f"📁 Results saved in: test_results/")

if __name__ == "__main__":
    main()
