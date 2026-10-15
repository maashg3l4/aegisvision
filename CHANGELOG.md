# Changelog

## [1.0.0-day1] - 2026-10-10
- Initial AegisVision V2 project foundation and workspace setup.
## [Wing 1.1] - Dataset Curation & YOLO Formatting
- Implemented XML to YOLO conversion script (`src/data/prepare_yolo_dataset.py`).
- Excluded `crazing_240` to perfectly match the historical 1,439 train / 360 val split.
- Generated `data.yaml` with 6 NEU-DET classes.
