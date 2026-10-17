# Changelog

## [1.0.0-day1] - 2026-10-10
- Initial AegisVision V2 project foundation and workspace setup.
## [Wing 1.1] - Dataset Curation & YOLO Formatting
- Implemented XML to YOLO conversion script (`src/data/prepare_yolo_dataset.py`).
- Excluded `crazing_240` to perfectly match the historical 1,439 train / 360 val split.
- Generated `data.yaml` with 6 NEU-DET classes.

## [Wing 2.10] - GC10-DET generalization setup
- Added `src/data/prepare_gc10_det.py` (PASCAL VOC -> YOLO, 8:1:1 split).
- Converted 2,292 records -> 1,830 train / 227 val / 235 test.
- Normalized 10_yaozhed -> 10_yaozhe; excluded garbage label 'd'.
- Dataset persisted to Drive with hash verification.

## [Phase 5] - Project Finalization & Scientific Reporting
- Documented successful NEU-DET routing (AUROC 0.82).
- Documented GC10-DET negative transfer result (AUROC 0.53).
- Documented Conformal Prediction failure at 22-shots.
- Generated comprehensive `docs/reports/final_report.md`.
