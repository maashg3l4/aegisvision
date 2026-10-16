"""AegisVision - GC10-DET to YOLO conversion.

Converts GC10-DET PASCAL VOC XMLs into YOLO format with a stratified
8:1:1 split (seed=42), matching the standard benchmark protocol.

Excludes:
  - Label 'd' (1 garbage annotation)
  - Images without annotations

Class normalization:
  - '10_yaozhed' merged into '10_yaozhe' (typo variant)

Usage:
    python prepare_gc10_det.py --raw <gc10_root> --out <yolo_root>
"""
import argparse, os, shutil, random, yaml
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import cv2

LABEL_MAP = {
    "1_chongkong": 0, "2_hanfeng": 1, "3_yueyawan": 2,
    "4_shuiban": 3, "5_youban": 4, "6_siban": 5,
    "7_yiwu": 6, "8_yahen": 7, "9_zhehen": 8,
    "10_yaozhe": 9, "10_yaozhed": 9,
}
EXCLUDED = {"d"}
CLASS_NAMES = [
    "punching_hole", "welding_line", "crescent_gap", "water_spot",
    "oil_spot", "silk_spot", "inclusion", "rolled_pit",
    "crease", "waist_folding",
]
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1


def build_image_index(root):
    idx = {}
    basename_map = defaultdict(list)
    for folder in os.listdir(root):
        fp = os.path.join(root, folder)
        if not os.path.isdir(fp) or folder == "lable":
            continue
        for f in os.listdir(fp):
            if f.endswith(".jpg"):
                base = os.path.splitext(f)[0]
                idx[(folder, base)] = os.path.join(fp, f)
                basename_map[base].append((folder, os.path.join(fp, f)))
    return idx, basename_map


def parse_records(root, lable_dir, image_idx, basename_map):
    records, unresolved = [], []
    for xf in sorted(os.listdir(lable_dir)):
        if not xf.endswith(".xml"):
            continue
        try:
            tree = ET.parse(os.path.join(lable_dir, xf))
        except Exception:
            continue
        r = tree.getroot()
        folder_el = r.find("folder")
        folder = folder_el.text.strip() if folder_el is not None else "?"
        fname_el = r.find("filename")
        if fname_el is None:
            continue
        base = os.path.splitext(fname_el.text.strip())[0]

        img_path = image_idx.get((folder, base))
        if img_path is None:
            cands = basename_map.get(base, [])
            if len(cands) == 1:
                img_path = cands[0][1]
        if img_path is None:
            unresolved.append((folder, base))
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        objs = []
        for o in r.findall("object"):
            n = o.find("name")
            if n is None or n.text.strip() in EXCLUDED:
                continue
            label = n.text.strip()
            if label not in LABEL_MAP:
                continue
            bb = o.find("bndbox")
            if bb is None:
                continue
            try:
                xmin = float(bb.find("xmin").text)
                ymin = float(bb.find("ymin").text)
                xmax = float(bb.find("xmax").text)
                ymax = float(bb.find("ymax").text)
            except Exception:
                continue
            if xmin >= xmax or ymin >= ymax:
                continue
            objs.append((LABEL_MAP[label], xmin, ymin, xmax, ymax, w, h))
        if objs:
            records.append((img_path, objs))
    return records


def split_records(records):
    random.seed(SEED)
    by_class = defaultdict(list)
    for rec in records:
        dom = Counter(o[0] for o in rec[1]).most_common(1)[0][0]
        by_class[dom].append(rec)

    train, val, test = [], [], []
    for _, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_tr = int(n * TRAIN_FRAC)
        n_v = int(n * VAL_FRAC)
        train.extend(items[:n_tr])
        val.extend(items[n_tr:n_tr + n_v])
        test.extend(items[n_tr + n_v:])
    return train, val, test


def write_split(records, split, out_root):
    for img_path, objs in records:
        base = os.path.splitext(os.path.basename(img_path))[0]
        shutil.copy2(img_path, os.path.join(out_root, "images", split, base + ".jpg"))
        lines = []
        for cid, xmin, ymin, xmax, ymax, w, h in objs:
            lines.append(
                f"{cid} {((xmin+xmax)/2)/w:.6f} {((ymin+ymax)/2)/h:.6f} "
                f"{(xmax-xmin)/w:.6f} {(ymax-ymin)/h:.6f}"
            )
        with open(os.path.join(out_root, "labels", split, base + ".txt"), "w") as f:
            f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    image_idx, basename_map = build_image_index(args.raw)
    records = parse_records(args.raw, os.path.join(args.raw, "lable"),
                            image_idx, basename_map)
    print(f"Valid records: {len(records)}")

    train, val, test = split_records(records)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    write_split(train, "train", args.out)
    write_split(val, "val", args.out)
    write_split(test, "test", args.out)

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.dump({
            "path": args.out,
            "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        }, f)
    print(f"data.yaml written to {args.out}/data.yaml")


if __name__ == "__main__":
    main()
