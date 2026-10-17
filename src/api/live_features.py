"""AegisVision - Live Feature Extractor for Production.
Extracts the 15 deployable prediction-statistics from a YOLO result.
(Excludes class_mismatch, which requires ground truth).
"""
import numpy as np
from scipy.stats import entropy as scipy_entropy

FEATURES = ['max_conf','mean_conf','std_conf','p90_conf','top1_gap','conf_entropy',
            'n_preds','n_mid','n_high','mean_area','min_area','mean_ar','std_ar',
            'small_ratio','unique_classes_pred']

def extract_live_features(yolo_results):
    """Takes a YOLO results object and returns a dict of 15 features."""
    boxes = yolo_results.boxes
    if len(boxes) == 0:
        # Fallback for completely empty images
        return {f: 0.0 for f in FEATURES}
        
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    xywh = boxes.xywh.cpu().numpy()
    
    areas = xywh[:, 2] * xywh[:, 3]
    ars = xywh[:, 2] / np.maximum(xywh[:, 3], 1e-6)
    
    hist, _ = np.histogram(scores, bins=10, range=(0, 1))
    
    return {
        'max_conf': float(scores.max()),
        'mean_conf': float(scores.mean()),
        'std_conf': float(scores.std()),
        'p90_conf': float(np.percentile(scores, 90)),
        'top1_gap': float(scores[0] - scores.mean()) if len(scores) > 1 else 0.0,
        'conf_entropy': float(scipy_entropy(hist + 1e-9)),
        'n_preds': len(scores),
        'n_mid': int(((scores > 0.2) & (scores <= 0.5)).sum()),
        'n_high': int((scores > 0.5).sum()),
        'mean_area': float(areas.mean()),
        'min_area': float(areas.min()),
        'mean_ar': float(ars.mean()),
        'std_ar': float(ars.std()),
        'small_ratio': float((areas < 500).sum()) / len(areas),
        'unique_classes_pred': int(len(set(classes)))
    }
