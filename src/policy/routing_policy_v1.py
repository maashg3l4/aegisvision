"""AegisVision Routing Policy v1.0.0

Frozen policy artifact for human-in-the-loop routing.
Companion file: routing_policy_v1.json (loaded at inference time).

Policy:
1. Route all images of weak classes (AUROC < 0.7) to human review.
2. For strong classes, apply conformal threshold lambda on LR danger score.

See: /content/drive/MyDrive/AegisVision/policies/routing_policy_v1.json
"""
# Feature order (MUST match training)
FEATURES = ['max_conf', 'mean_conf', 'std_conf', 'p90_conf', 'top1_gap', 'conf_entropy', 'n_preds', 'n_mid', 'n_high', 'mean_area', 'min_area', 'mean_ar', 'std_ar', 'small_ratio', 'unique_classes_pred', 'class_mismatch']
WEAK_CLASSES_HARD_ROUTE = []
CONFORMAL_LAMBDA = 0.011443492059197863
ALPHA = 0.1
