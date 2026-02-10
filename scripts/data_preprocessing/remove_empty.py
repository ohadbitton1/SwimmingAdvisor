import os
from pathlib import Path
from tqdm import tqdm

# Path to dataset
BASE_PATH = Path("/content/datasets")

def purge_empty_labels():
    print("Starting cleanup (Scanning for empty labels without training)...")
    
    # 1. Collect all text files
    all_label_files = list(BASE_PATH.rglob('*.txt'))
    
    if not all_label_files:
        print("No label files found.")
        return

    deleted_count = 0
    kept_count = 0
    
    # Progress bar
    progress_bar = tqdm(all_label_files, desc="Checking files")
    
    for label_file in progress_bar:
        is_empty = False
        
        # Double check: Is file size 0 or content empty?
        if label_file.stat().st_size == 0:
            is_empty = True
        else:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    is_empty = True
        
        if is_empty:
            # Delete label file
            label_file.unlink()
            
            # Delete corresponding image
            # Try finding image in ../images or same folder
            img_dir = label_file.parent.parent / 'images'
            if not img_dir.exists(): 
                img_dir = label_file.parent

            # Check all extensions
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                img_path = img_dir / (label_file.stem + ext)
                if img_path.exists():
                    img_path.unlink() # Delete image
                    break
            
            deleted_count += 1
            progress_bar.set_postfix(deleted=deleted_count)
        else:
            kept_count += 1

    # --- Final Report ---
    print(f"\n" + "="*40)
    print(f"       DATA CLEANUP REPORT       ")
    print(f"="*40)
    print(f"Total files scanned: {len(all_label_files)}")
    print(f"Deleted (Empty):     {deleted_count}")
    print(f"Remaining (Good):    {kept_count}")
    print(f"="*40)
    print("Dataset is clean. No training performed.")

if __name__ == "__main__":
    purge_empty_labels()