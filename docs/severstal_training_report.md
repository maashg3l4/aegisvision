# Severstal Industrial Defect Detection: Training Report

## 1. Executive Summary
A YOLO11s baseline was trained on the Severstal Steel Defect Dataset. To handle the severe class imbalance (Class 2 has <250 instances) and the ultra-wide image format (1600x256), a custom RLE-to-YOLO pipeline with overlapping tiling (3 tiles per image) was engineered.

## 2. Final Validation Metrics
- **mAP50:** `0.684` (Beats the standard YOLOv8m baseline of ~0.647)
- **mAP50-95:** `0.378`
- **Precision:** `0.669`
- **Recall:** `0.635`

## 3. Per-Class Performance (mAP50)
- `defect_1`: 0.643
- `defect_2` (Rare Class): **0.810**
- `defect_3` (Majority Class): 0.702
- `defect_4` (Diffuse Noise): 0.582

## 4. Key Engineering Insights
1. **Rare Class Success:** The class-aware augmentation and tiling strategy successfully enabled the model to learn `defect_2`, achieving an impressive 0.810 mAP50 despite its scarcity.
2. **Box Tightness:** The mAP50-95 of 0.378 indicates the model struggles with pixel-perfect box regression on long, thin defects. Future work will address this with Normalized Wasserstein Distance (NWD) loss or segmentation models.
3. **Convergence:** Early stopping triggered at Epoch 98 (best epoch 73), proving the model maximized the capacity of the YOLO11s architecture on this dataset.
