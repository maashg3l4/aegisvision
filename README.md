# AegisVision

## Overview
**AegisVision** is a **Confidence-Aware, Human-in-the-Loop Industrial Defect Inspection Assistant** designed for industrial steel surface defect detection.

## Mission
"AI for the Inspector, not AI for the Researcher."

## Core Engineering Principle
"Push performance as high as legitimately possible — without sacrificing scientific validity, generalization, robustness, reproducibility, or auditability."

## System Goal
- Automate when evidence is sufficient.
- Escalate when uncertainty matters.
- Assist humans with actionable evidence rather than pretending every prediction is correct.

## Historical Verified Reference Baseline (NEU-DET V2)
- **Precision**: 0.6907
- **Recall**: 0.6904
- **mAP50**: 0.7425
- **mAP50-95**: 0.3988
*(Note: These are historical verified references to be independently reproduced in a later phase of AegisVision V2).*

## Repository Structure
- `configs/`: Dataset, evaluation, training, and experiment configurations.
- `data/`: Local manifests, splits, and schemas.
- `src/aegisvision/`: Core modular pipeline (common, data, detection, evaluation, uncertainty, verification, human_review).
- `docs/`: Architecture, evaluation protocol, experiment policy, risk register, and system card.
- `notebooks/`: Step-by-step experiment and audit notebooks.
- `tests/`: Unit, integration, and regression tests.

## License
MIT License
