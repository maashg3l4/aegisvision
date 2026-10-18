# Severstal Training Progress Log

## Date: 2026-10-19

### Completed Today:
1. **Dataset Conversion:** Successfully converted the 12,568 Severstal images into 18,000+ YOLO-formatted tiles.
2. **Stratified Splitting:** Implemented image-level splitting (5999 Train / 333 Val / 334 Test) to prevent data leakage.
3. **RLE Decoding:** Verified the `rle2mask` and `mask2bbox` functions against ground truth visualizations.
4. **Baseline Training:** Initiated YOLO11s training. By epoch 6, the model achieved a preliminary mAP50 of 0.33.
5. **Safe Persistence:** Successfully saved `last.pt` and `best.pt` checkpoints to Google Drive before the Colab GPU quota limit was reached.

### Next Steps (Pending GPU Quota Reset):
- Resume training from `last.pt` at Epoch 7.
- Target mAP50 > 0.60 to match the published YOLOv8m baseline.
