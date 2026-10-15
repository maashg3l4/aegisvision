# Wing 1.2: Frozen YOLO11s Baseline Report

## Training Configuration
- Model: YOLO11s
- Dataset: NEU-DET (1,439 train / 360 val)
- Epochs: 100 (Early stopping patience: 20)
- Batch Size: 16
- Image Size: 640
- Seed: 42 (Deterministic)
- Hardware: Tesla T4 GPU

## Final Validation Metrics (conf=0.001, IoU=0.70)
| Metric | Value |
| :--- | :--- |
| Precision | 0.688 |
| Recall | 0.694 |
| mAP50 | 0.757 |
| mAP50-95 | 0.405 |

## Per-Class Performance (mAP50)
- crazing: 0.544
- inclusion: 0.813
- patches: 0.916
- pitted_surface: 0.826
- rolled-in_scale: 0.585
- scratches: 0.857

## Key Finding
The baseline exceeds the historical reference (mAP50: 0.7425 -> 0.757). 
Significant weaknesses remain in `crazing` and `rolled-in_scale`. 
These classes will be the primary focus of the uncertainty and calibration phases.
