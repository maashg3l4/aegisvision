# AegisVision Final Scientific Report (Phase 1 - 4)

## 1. Executive Summary
AegisVision successfully built a Confidence-Aware, Human-in-the-Loop routing system. The system achieves state-of-the-art routing performance on the source domain (NEU-DET), but rigorous testing reveals that post-hoc prediction statistics fail to generalize zero-shot across severe industrial domain shifts (NEU-DET to GC10-DET) without target-domain calibration data.

## 2. Phase 1 & 2: NEU-DET Baseline & Routing (Success)
- **Detector:** YOLO11s achieved a verified mAP50 of 0.757 on NEU-DET.
- **Routing Policy:** A 16-feature Logistic Regression model achieved an OOF AUROC of 0.8213.
- **Operational Impact:** At a 20% human review budget, the policy reduced residual risk from 28.4% (max-confidence baseline) to 13.8%.

## 3. Phase 3: Cross-Dataset Generalization (GC10-DET)
- **Detector:** YOLO11s achieved a mAP50 of 0.656 on GC10-DET (beating standard literature baselines of 60.4%).
- **Zero-Shot Routing Transfer:** The NEU-DET policy failed to transfer (AUROC 0.5365). Deep YOLO embeddings also failed (AUROC 0.47).
- **Few-Shot Adaptation:** Using a Cosine Prototype classifier with only 22 target images recovered the AUROC to 0.6254.

## 4. Phase 4: Conformal Prediction & Safety Guarantees
- **The 10% Risk Guarantee:** Transductive Conformal Prediction using 22 images failed to maintain a 10% risk bound on the accepted set.
- **Conclusion:** 22 calibration images are mathematically insufficient to guarantee a 10% risk bound under severe domain shift (resolution and texture change). The minimum viable calibration set is estimated at 90+ images.

## 5. Core Scientific Contributions
1.  A fully reproducible pipeline for defect detection and uncertainty routing.
2.  Empirical proof that confidence-based routing fails under domain shift.
3.  A novel benchmark showing that simple prediction statistics (entropy, density) outperform deep semantic embeddings for zero-shot uncertainty estimation in industrial vision.

## 6. Future Work (Phase 5)
- Integrate the Aegis Report Assistant (Conversational XAI) for human-readable audit logs.
- Test adaptive calibration methods on larger target datasets (e.g., Severstal).
- Deploy the FastAPI + Streamlit prototype for live factory simulation.
