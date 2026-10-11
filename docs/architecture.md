# AegisVision Architecture

AegisVision V2 follows a modular, production-oriented pipeline architecture designed around reliability-aware inspection and human-in-the-loop escalation.

## Core Modules (`src/aegisvision/`)
1. **common/**: Shared utilities, logging, and configuration loaders.
2. **data/**: Dataset loading, manifest validation, and split management.
3. **detection/**: Core YOLO-based object detection wrappers and inference engines.
4. **evaluation/**: Standardized metric computation (Precision, Recall, mAP50, mAP50-95).
5. **uncertainty/**: Confidence calibration, uncertainty estimation, and failure mode detection.
6. **verification/**: Evidence grounding and sanity checks.
7. **human_review/**: Human-in-the-loop escalation logic for ambiguous or low-confidence predictions.
