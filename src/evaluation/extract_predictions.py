"""
AegisVision - Wing 1.3: Baseline Error Analysis & Confidence Extraction
Runs the baseline model on the validation set and saves per-image predictions.
"""
import os
import pandas as pd
from ultralytics import YOLO

def main():
    model_path = "/content/drive/MyDrive/AegisVision/runs/baseline_yolo11s/weights/best.pt"
    data_yaml = "/content/drive/MyDrive/AegisVision/data/processed/neu_det_yolo/data.yaml"
    output_dir = "/content/drive/MyDrive/AegisVision/reports/baseline"
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    # Run validation with low confidence to capture all predictions for calibration
    results = model.val(data=data_yaml, conf=0.001, iou=0.70, save_json=True, project=output_dir, name="val_predictions")
    print(f"Validation complete. Results saved to {results.save_dir}")

if __name__ == "__main__":
    main()
