# Routing Policy v2.0.0 - Documentation

## Formulation
Budget-constrained ranking. Given a review budget B%, route top-B% highest-danger images to humans.

## Why not conformal guarantee?
Strict conformal risk control at alpha=0.10 required accepting only 2.5% of images on a 360-image calibration set. This is mathematically correct but operationally useless. We switched to the fixed-budget formulation, which yields usable operating points.

## Operating Points (from 5-fold OOF CV on 360 val images)
| Risk Target | Review Budget | Expected Residual Risk |
| :--- | :--- | :--- |
| <= 0.05 | 59% | 0.0473 |
| <= 0.08 | 50% | 0.0778 |
| <= 0.10 | 43% | 0.0976 |
| <= 0.15 | 15% | 0.1498 |
| <= 0.20 |  4% | 0.2000 |

## Artifacts
- `src/policy/routing_policy.py` - inference module
- `/content/drive/MyDrive/AegisVision/policies/routing_policy_v2.json` - frozen model parameters

## Caveats
- Empirical results, not statistical guarantees.
- Small calibration set (360 images).
- Generalization must be validated on independent data (Wing 2.10).
