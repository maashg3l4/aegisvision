"""
AegisVision - Wing 1.1: Dataset Curation & YOLO Formatting
Converts the raw NEU-DET XML annotations into YOLO format.
Excludes 'crazing_240' to match the historical 1439/360 train/val split.
"""
import os
import shutil
import xml.etree.ElementTree as ET
import cv2
import yaml

RAW_BASE = "/root/.cache/kagglehub/datasets/kaustubhdikshit/neu-surface-defect-database/versions/1/NEU-DET"
DRIVE_BASE = "/content/drive/MyDrive/AegisVision/data/processed/neu_det_yolo"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

def convert_xml_to_yolo(xml_path, img_width, img_height):
    lines = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_TO_ID:
            continue
        cls_id = CLASS_TO_ID[cls_name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines

def main():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DRIVE_BASE, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DRIVE_BASE, "labels", split), exist_ok=True)

    for split, target_split in [("train", "train"), ("validation", "val")]:
        img_dir = os.path.join(RAW_BASE, split, "images")
        ann_dir = os.path.join(RAW_BASE, split, "annotations")
        processed_count = 0
        skipped_count = 0
        
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                base_name = os.path.splitext(f)[0]
                img_path = os.path.join(root, f)
                ann_path = os.path.join(ann_dir, base_name + ".xml")
                
                if base_name == "crazing_240":
                    print(f"  [Excluded] {base_name} - missing annotation")
                    skipped_count += 1
                    continue
                if not os.path.exists(ann_path):
                    skipped_count += 1
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    skipped_count += 1
                    continue
                h, w = img.shape[:2]
                
                yolo_lines = convert_xml_to_yolo(ann_path, w, h)
                if not yolo_lines:
                    skipped_count += 1
                    continue
                    
                shutil.copy2(img_path, os.path.join(DRIVE_BASE, "images", target_split, f))
                with open(os.path.join(DRIVE_BASE, "labels", target_split, base_name + ".txt"), "w") as out_f:
                    out_f.write("\n".join(yolo_lines))
                processed_count += 1
        print(f"[{split.capitalize()}] Processed: {processed_count}, Skipped/Excluded: {skipped_count}")

    yaml_path = os.path.join(DRIVE_BASE, "data.yaml")
    data_config = {
        "path": DRIVE_BASE,
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)}
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print(f"Dataset creation complete. data.yaml written to: {yaml_path}")

if __name__ == "__main__":
    main()
