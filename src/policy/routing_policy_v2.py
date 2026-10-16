"""AegisVision Routing Policy v2.0.0 — Budget-Constrained Ranking

Formulation: given review budget B%, route top-B% highest-danger images to humans.
Companion: /content/drive/MyDrive/AegisVision/policies/routing_policy_v2.json
"""
FEATURES = ['max_conf', 'mean_conf', 'std_conf', 'p90_conf', 'top1_gap', 'conf_entropy', 'n_preds', 'n_mid', 'n_high', 'mean_area', 'min_area', 'mean_ar', 'std_ar', 'small_ratio', 'unique_classes_pred', 'class_mismatch']
OPERATING_POINTS = {
  "0.05": 0.5888888888888889,
  "0.08": 0.5,
  "0.1": 0.4305555555555556,
  "0.15": 0.14722222222222225,
  "0.2": 0.04166666666666663
}
