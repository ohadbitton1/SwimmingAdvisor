import os
import shutil
from roboflow import Roboflow

# ==========================================
# CONFIGURATION
# ==========================================
# Replace these with your actual Roboflow credentials
API_KEY = "ROBOFLOW_API_KEY"
WORKSPACE_NAME = "swimadvisor"

# Target directory for all datasets
BASE_DATA_PATH = "/content/datasets"

# Dataset Registry: Map your local names to Roboflow Project IDs and Versions
# gen1 = The original 'augmented' dataset
# gen2 = The new 30-image dataset (expanded)
# gen3 = Nunnari Labs (large)
# gen4 = Nunnari Labs (small/swimAI 2)
DATASET_REGISTRY = {
    "gen1": {"project_id": "gen1-5tjbt", "version": 2},
    "gen2": {"project_id": "gen2-umtgg", "version": 2},
    "gen3": {"project_id": "gen3-jg0kj", "version": 1},
    "gen4": {"project_id": "gen4-w2hgh", "version": 1},
}
# =========================================
def download_and_organize():
    """
    Downloads datasets from Roboflow if they don't exist locally,
    and renames them to a standard format (gen1, gen2, etc.).
    """
    print(f"[*] Initializing Roboflow with workspace: {WORKSPACE_NAME}")
    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace(WORKSPACE_NAME)

    # Ensure base directory exists
    if not os.path.exists(BASE_DATA_PATH):
        os.makedirs(BASE_DATA_PATH)
        print(f"[*] Created base directory: {BASE_DATA_PATH}")

    for gen_name, info in DATASET_REGISTRY.items():
        target_dir = os.path.join(BASE_DATA_PATH, gen_name)
        
        # 1. Hybrid Check: Skip if already exists
        if os.path.exists(target_dir):
            print(f"[~] Dataset {gen_name} already exists at {target_dir}. Skipping download.")
            continue

        print(f"[+] Downloading {gen_name} (Project: {info['project_id']}, Version: {info['version']})...")

        try:
            # 2. Download from API
            project = workspace.project(info["project_id"])
            version = project.version(info["version"])
            dataset = version.download("yolov8")

            # 3. Organize: Move from Roboflow's random name to our standard name
            downloaded_path = dataset.location
            
            # Move the content to our target directory
            shutil.move(downloaded_path, target_dir)
            
            # 4. Cleanup: Remove the parent folder created by Roboflow if it's empty or clutter
            # (Roboflow often downloads into folder/Project-Name-Version)
            parent_dir = os.path.dirname(downloaded_path)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)

            print(f"[V] Successfully downloaded and organized: {gen_name}")

        except Exception as e:
            print(f"[!] Error processing {gen_name}: {e}")

if __name__ == "__main__":
    download_and_organize()