"""AegisVision - Image-Level Feature Engineering.

Reads predictions.json + instances_val.json and produces a per-image CSV
with 17 features used by the routing policy.

Usage:
    python build_features.py --preds predictions.json --gt instances_val.json --out features.csv
"""
import argparse
import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

HIGH_CONF = 0.5


def _iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def compute_features(preds, gt):
    id2name = {im["id"]: im["file_name"] for im in gt["images"]}
    name2id = {im["file_name"]: im["id"] for im in gt["images"]}

    preds_by_img = defaultdict(list)
    for p in preds:
        base = os.path.splitext(p["file_name"])[0] + ".jpg"
        if base in name2id:
            preds_by_img[name2id[base]].append(p)

    gt_by_img = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_img[ann["image_id"]].append(ann)

    rows = []
    for img_id, gt_anns in gt_by_img.items():
        img_preds = sorted(preds_by_img.get(img_id, []),
                           key=lambda p: -p["score"])
        fname = id2name[img_id]

        matched = set()
        n_tp = n_high_fp = n_low_fp = 0
        for p in img_preds:
            best_iou, best_gt = 0.0, None
            for g in gt_anns:
                if g["id"] in matched:
                    continue
                if g["category_id"] != p["category_id"]:
                    continue
                i = _iou(p["bbox"], g["bbox"])
                if i > best_iou:
                    best_iou, best_gt = i, g
            if best_iou >= 0.5 and best_gt is not None:
                matched.add(best_gt["id"])
                n_tp += 1
            elif p["score"] > HIGH_CONF:
                n_high_fp += 1
            else:
                n_low_fp += 1
        n_fn = len(gt_anns) - len(matched)
        dangerous = int((n_high_fp > 0) or (n_fn > 0))

        scores = np.array([p["score"] for p in img_preds]) if img_preds else np.array([0.0])
        hist, _ = np.histogram(scores, bins=10, range=(0, 1))
        conf_entropy = float(scipy_entropy(hist + 1e-9))

        if img_preds:
            areas = np.array([p["bbox"][2] * p["bbox"][3] for p in img_preds])
            ars = np.array([p["bbox"][2] / max(p["bbox"][3], 1e-6) for p in img_preds])
        else:
            areas = np.array([0.0])
            ars = np.array([0.0])

        pred_classes = [p["category_id"] for p in img_preds]
        gt_classes = set(g["category_id"] for g in gt_anns)

        rows.append({
            "image_id": img_id,
            "file_name": fname,
            "class_prefix": fname.split("_")[0],
            "n_gt": len(gt_anns),
            "n_tp": n_tp, "n_high_fp": n_high_fp,
            "n_low_fp": n_low_fp, "n_fn": n_fn,
            "has_dangerous_error": dangerous,
            "max_conf": float(scores.max()),
            "mean_conf": float(scores.mean()),
            "std_conf": float(scores.std()),
            "p90_conf": float(np.percentile(scores, 90)),
            "top1_gap": float(scores[0] - scores.mean()) if len(scores) > 1 else 0.0,
            "conf_entropy": conf_entropy,
            "n_preds": len(img_preds),
            "n_high": int((scores > HIGH_CONF).sum()),
            "n_mid": int(((scores > 0.2) & (scores <= 0.5)).sum()),
            "high_ratio": float((scores > HIGH_CONF).sum()) / max(len(scores), 1),
            "mean_area": float(areas.mean()),
            "min_area": float(areas.min()),
            "mean_ar": float(ars.mean()),
            "std_ar": float(ars.std()),
            "small_ratio": float((areas < 500).sum()) / len(areas),
            "unique_classes_pred": len(set(pred_classes)),
            "class_mismatch": len(set(pred_classes) - gt_classes),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.preds) as f:
        preds = json.load(f)
    with open(args.gt) as f:
        gt = json.load(f)

    df = compute_features(preds, gt)
    df.to_csv(args.out, index=False)
    print(f"[features] wrote {args.out} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
