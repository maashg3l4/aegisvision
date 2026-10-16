"""AegisVision Routing Policy v2.0.0 - Budget-Constrained Ranking.

At inference:
    score = danger_score(feature_dict, policy)          # float in [0, 1]
    review_idx = route_batch(list_of_scores, budget)    # indices to review

Design rationale:
    Strict conformal guarantee at risk <= 10% failed on 360-image calibration.
    Switched to fixed-budget formulation: given budget B%, rank images by
    danger score and route top-B% to human review.
"""
import json
import numpy as np

FEATURES = [
    "max_conf", "mean_conf", "std_conf", "p90_conf", "top1_gap", "conf_entropy",
    "n_preds", "n_mid", "n_high", "mean_area", "min_area", "mean_ar", "std_ar",
    "small_ratio", "unique_classes_pred", "class_mismatch",
]

OPERATING_POINTS = {
    "risk_target_0.05": {"review_budget": 0.59, "expected_residual_risk": 0.0473},
    "risk_target_0.08": {"review_budget": 0.50, "expected_residual_risk": 0.0778},
    "risk_target_0.10": {"review_budget": 0.43, "expected_residual_risk": 0.0976},
    "risk_target_0.15": {"review_budget": 0.15, "expected_residual_risk": 0.1498},
    "risk_target_0.20": {"review_budget": 0.04, "expected_residual_risk": 0.2000},
}


def load_policy(path):
    """Load a frozen policy JSON."""
    with open(path) as f:
        return json.load(f)


def danger_score(feature_dict, policy):
    """Compute the logistic-regression danger probability for a single image."""
    x = np.array([feature_dict[f] for f in policy["features"]], dtype=float)
    mu = np.array(policy["scaler"]["mean"])
    sd = np.array(policy["scaler"]["scale"])
    w = np.array(policy["lr_model"]["coefficients"])
    b = policy["lr_model"]["intercept"]
    z = float(((x - mu) / sd) @ w + b)
    return 1.0 / (1.0 + np.exp(-z))


def route_batch(danger_scores, budget_fraction):
    """Given scores for a batch, return the indices that should be reviewed.

    Review budget_fraction * N images with the HIGHEST danger scores.
    """
    arr = np.asarray(danger_scores, dtype=float)
    n = len(arr)
    k = max(1, int(budget_fraction * n))
    order = np.argsort(-arr)
    return set(order[:k].tolist())
