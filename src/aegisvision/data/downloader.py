# AegisVision V2 — Robust Dataset Ingestion Handler
import os
from roboflow import Roboflow

def download_neudet_dataset(api_key: str):
    print("Initializing Roboflow client with verified public slugs...")
    try:
        rf = Roboflow(api_key=api_key)
        # Using verified public workspace and project slugs
        proj = rf.workspace("park-sung-ho").project("neu-det-object-detection")
        version = proj.version(1)
        
        target_dir = "/content/drive/MyDrive/AegisVision/data/raw"
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Downloading YOLO-formatted NEU-DET dataset to {target_dir}...")
        dataset = version.download("yolov8", location=target_dir)
        print("Dataset successfully downloaded and extracted!")
        return True
    except Exception as e:
        print(f"[ERROR] Automated download failed: {e}")
        return False
