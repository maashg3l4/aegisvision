"""AegisVision - Prediction & Ground Truth Extraction.

Runs YOLO validation on the val split with low confidence, saving:
  - predictions.json  (every predicted box with score)
  - instances_val.json (COCO ground truth reconstructed from YOLO labels)

Usage:
    python extract_predictions.py --model best.pt --data data.yaml --out ./out
"""
import argparse
import json
import os
import cv2
from ultralytics import YOLO

CLASSES = ["crazing", "inclusion", "patches",
           "pitted_surface", "rolled-in_scale", "scratches"]


def reconstruct_gt_coco(images_dir, labels_dir, classes):
    """Rebuild COCO-format GT JSON from YOLO-format label files."""
    gt_images, gt_anns = [], []
    ann_id, img_id = 1, 1
    for f in sorted(os.listdir(images_dir)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(images_dir, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_images.append({"id": img_id, "file_name": f, "width": w, "height": h})
        lbl = os.path.join(labels_dir, base + ".txt")
        if os.path.exists(lbl):
            for line in open(lbl):
                p = line.strip().split()
                if len(p) < 5:
                    continue
                cid = int(p[0])
                xc, yc, bw, bh = map(float, p[1:5])
                box = [(xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h]
                gt_anns.append({"id": ann_id, "image_id": img_id,
                                "category_id": cid + 1, "bbox": box,
                                "area": box[2] * box[3], "iscrowd": 0})
                ann_id += 1
        img_id += 1
    return {"images": gt_images, "annotations": gt_anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(classes)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.70)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.model)
    results = model.val(data=args.data, conf=args.conf, iou=args.iou,
                        split="val", save_json=True,
                        project=args.out, name="val_run", exist_ok=True)

    preds_src = os.path.join(results.save_dir, "predictions.json")
    preds_dst = os.path.join(args.out, "predictions.json")
    if os.path.exists(preds_src) and preds_src != preds_dst:
        os.replace(preds_src, preds_dst)

    data_dir = os.path.dirname(args.data)
    gt = reconstruct_gt_coco(
        os.path.join(data_dir, "images", "val"),
        os.path.join(data_dir, "labels", "val"),
        CLASSES,
    )
    with open(os.path.join(args.out, "instances_val.json"), "w") as f:
        json.dump(gt, f)

    print(f"[extract] predictions.json -> {preds_dst}")
    print(f"[extract] instances_val.json -> {args.out} "
          f"({len(gt['images'])} imgs, {len(gt['annotations'])} boxes)")


if __name__ == "__main__":
    main()
