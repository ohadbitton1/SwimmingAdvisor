import os
import glob
import yaml
import shutil
import torch
from ultralytics import YOLO
from tqdm import tqdm

DATASET_ROOT = "/content/datasets"
MODEL_NAME = 'yolov8x-pose.pt' 
CONF_THRESHOLD = 0.15

# Point filter (excludes eyes)
KEEP_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def clean_old_labels():
    print("[INFO] Cleaning ALL old label files...")
    txt_files = glob.glob(f"{DATASET_ROOT}/**/*.txt", recursive=True)
    for f in txt_files:
        if not f.endswith('classes.txt') and 'README' not in f:
            try: os.remove(f)
            except: pass

def auto_label_images():
    clean_old_labels()
    print(f"[INFO] Starting Auto-Labeling (13 Points + SINGLE SWIMMER ONLY)...")
    
    if not os.path.exists(DATASET_ROOT): return
    model = YOLO(MODEL_NAME)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(f"{DATASET_ROOT}/**/{ext}", recursive=True))

    for img_path in tqdm(image_files, desc="Labeling"):
        label_path = os.path.splitext(img_path)[0] + ".txt"
        
        # Run inference
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        
        lines = []
        # --- The big change: take only the best result! ---
        if results.boxes.conf.numel() > 0:
            # Find the index of the swimmer with the highest confidence
            best_idx = results.boxes.conf.argmax().item()
            
            # Take only that specific detection
            kpts = results.keypoints.xy[best_idx].cpu().numpy()
            box = results.boxes.xywhn[best_idx].cpu().numpy()
            h, w = results.orig_shape

            kpt_line = []
            for idx in KEEP_INDICES:
                if idx < len(kpts):
                    kx, ky = kpts[idx]
                    if kx == 0 and ky == 0: kpt_line.extend([0.0, 0.0, 0.0])
                    else: kpt_line.extend([kx/w, ky/h, 2.0])
                else:
                    kpt_line.extend([0.0, 0.0, 0.0])
            
            lines.append(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} " + " ".join(map(str, kpt_line)))
        
        with open(label_path, 'w') as f: f.write("\n".join(lines))
    
    print("[INFO] Labels generated (Single swimmer per image).")

def create_unified_yaml():
    print("[INFO] Creating unified_13pt.yaml...")
    train_dirs = glob.glob(f"{DATASET_ROOT}/**/train/images", recursive=True)
    val_dirs = glob.glob(f"{DATASET_ROOT}/**/valid/images", recursive=True)
    test_dirs = glob.glob(f"{DATASET_ROOT}/**/test/images", recursive=True)
    if not test_dirs: test_dirs = val_dirs
    
    yaml_data = {
        'path': DATASET_ROOT, 'train': train_dirs, 'val': val_dirs, 'test': test_dirs,
        'names': {0: 'swimmer'},
        'kpt_shape': [13, 3], 
        'flip_idx': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
    }
    
    with open(f"{DATASET_ROOT}/unified_13pt.yaml", 'w') as f: yaml.dump(yaml_data, f, sort_keys=False)
    print(f"[SUCCESS] YAML saved.")

if __name__ == "__main__":
    try: import ultralytics
    except ImportError: os.system('pip install ultralytics -q')
    auto_label_images()
    create_unified_yaml()