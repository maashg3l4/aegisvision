# Evaluation Protocol

## Frozen Detector Protocol
To ensure strict reproducibility across experiments, detector evaluations must adhere to:
- Confidence threshold (`conf`): `0.001`
- Intersection over Union (`IoU`): `0.70`
- Image size (`imgsz`): `640`
- Evaluation split: `val` (NEU-DET V2 validation set)

## Historical Reference Baselines
- Precision = 0.6907
- Recall = 0.6904
- mAP50 = 0.7425
- mAP50-95 = 0.3988

*Note: These values serve as historical references and will be independently reproduced and audited in subsequent project phases.*
