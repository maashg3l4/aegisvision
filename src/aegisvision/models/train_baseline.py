# AegisVision V2 — YOLOv11 Baseline Training Script

import os
from ultralytics import YOLO

def train_baseline():
    print("Initializing YOLOv11 nano model for NEU-DET V2 baseline...")
    # Load pre-trained nano weights
    model = YOLO("yolo11n.pt")

    print("Starting training with frozen evaluation protocol...")
    results = model.train(
        data="configs/dataset/neu_det_v2.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.70,
        device=0,
        project="aegisvision_runs",
        name="yolov11_baseline"
    )
    print("Training complete! Results saved under aegisvision_runs/yolov11_baseline")

if __name__ == "__main__":
    train_baseline()
