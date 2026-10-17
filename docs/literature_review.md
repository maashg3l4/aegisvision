# AegisVision: Literature Review & Novelty Positioning

## 1. The Core Research Gap
Prior studies have explored conformal prediction, selective prediction, uncertainty-aware referral, and human-in-the-loop decision support in classification, solar-cell inspection, clinical screening, and safety-critical industrial systems. 

However, **few works combine**:
1. Training-free post-hoc routing.
2. Budget-constrained human review (fixed review budgets).
3. Steel-surface defect object detection (NEU-DET, GC10-DET).
4. Systematic cross-domain calibration analysis.

**Therefore, AegisVision is positioned as a training-free, budget-constrained, confidence-aware referral framework for steel-surface defect detection, with an explicit analysis of when confidence-based routing fails under domain shift and how much target-domain calibration data is required for statistical safety.**

## 2. Comparison Matrix (Key Papers)

| # | Paper | Method | Routing Strategy | Guarantee Type | Limitation vs. AegisVision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Angelopoulos & Bates (2021) | Conformal prediction framework | Set-size-based abstention | Conformal (distribution-free) | Foundational theory; no defect detection or domain-shift experiments. |
| 2 | Angelopoulos et al. (2020) | Conformal prediction sets for image classifiers | Defer low-confidence/ambiguous | Conformal coverage guarantee | Image-level classification only; not object detection. |
| 3 | Angelopoulos et al. (2022) | Conformal Risk Control | Controls user-defined risk | Conformal risk control | Risk function is task-specific; not tailored to human review budgets. |
| 4 | Selective Conformal Risk Control (2025) | Abstention-aware CRC | Abstain on high uncertainty | Conformal guarantee | General-purpose; requires adaptation for object detection routing. |
| 5 | Thomsen et al. (2023) | CNN + Conformal Prediction | Route low-confidence to human | Conformal | Solar cells (not steel); classification rather than object detection. |
| 6 | Singh et al. (2026) | Mondrian CP + cost-controlled | Cost-aware routing | Conformal + cost-based | Tabular data focus; not visual defect detection. |
| 7 | Geifman & El-Yaniv (2019) | Selective classification evaluation | Confidence-threshold rejection | Empirical risk-coverage | Does not address domain shift or conformal guarantees. |
| 8 | LDS-Net (2026) | Lightweight YOLO-style detector | None (fully automatic) | Empirical mAP | No uncertainty routing, human deferral, or statistical guarantee. |
| 9 | MSFE-YOLO (2025) | YOLO-based cross-domain evaluation | Simple confidence threshold | Empirical | Focuses on mAP; limited treatment of calibrated risk or safety. |

## 3. Novelty Statement for the Paper
Prior work has explored conformal prediction and human referral in classification and medical domains. However, a **training-free, budget-constrained routing framework for steel-surface defect detection**—with a **systematic analysis of cross-domain calibration requirements**—remains underexplored. 

AegisVision contributes:
1. A 16-feature prediction-statistics model that outperforms deep semantic embeddings for zero-shot uncertainty estimation.
2. An empirical proof that 22 images are insufficient for a 10% risk guarantee under severe domain shift.
3. A deployable budget-constrained ranking policy that reduces residual risk by 51% on in-domain data.
