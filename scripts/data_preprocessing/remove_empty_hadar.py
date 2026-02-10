import os
from pathlib import Path
from tqdm import tqdm

# Pointing EXACTLY to the labels subfolder
# This ensures we don't scan other datasets by mistake
BASE_DIR = Path("data/hadar_check")
LABEL_DIR = BASE_DIR / "labels"
IMAGE_DIR = BASE_DIR / "images"

def purge_hadar_check_only():
    print(f"[*] Targeting: {LABEL_DIR}")
    
    if not LABEL_DIR.exists():
        print(f"[!] Path not found: {LABEL_DIR}")
        return

    # Use .glob instead of .rglob to avoid recursive scanning of other folders
    label_files = list(LABEL_DIR.glob("*.txt"))
    
    # We ignore classes.txt if it exists
    label_files = [f for f in label_files if f.name != "classes.txt"]
    
    if not label_files:
        print("[?] No labels found in hadar_check/labels.")
        return

    deleted = 0
    pbar = tqdm(label_files, desc="Cleaning hadar_check")

    for lb_file in pbar:
        # Check if empty
        if lb_file.stat().st_size == 0:
            should_delete = True
        else:
            with open(lb_file, 'r') as f:
                if not f.read().strip():
                    should_delete = True
                else:
                    should_delete = False
        
        if should_delete:
            # 1. Delete Label
            lb_file.unlink()
            
            # 2. Delete corresponding Image
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                img_path = IMAGE_DIR / (lb_file.stem + ext)
                if img_path.exists():
                    img_path.unlink()
                    break
            
            deleted += 1
            pbar.set_postfix(deleted=deleted)

    print(f"\n{'='*30}")
    print(f"Scan Complete for hadar_check")
    print(f"Total Scanned: {len(label_files)}")
    print(f"Deleted:       {deleted}")
    print(f"Remaining:     {len(label_files) - deleted}")
    print(f"{'='*30}")

if __name__ == "__main__":
    purge_hadar_check_only()