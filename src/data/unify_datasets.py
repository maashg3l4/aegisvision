"""
AegisVision - Dataset Unification Engine.
Merges NEU-DET (6 classes) and GC10-DET (10 classes) into a single 16-class YOLO dataset.
"""
import os
import shutil
import yaml

# Define the master taxonomy
UNIFIED_CLASSES = [
    # NEU-DET Classes (0-5)
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
    # GC10-DET Classes (6-14) - Note: 'inclusion' overlaps, so we map it to the same ID
    "punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot", 
    "silk_spot", "rolled_pit", "crease", "waist_folding"
]

# Mapping from original dataset IDs to Unified IDs
NEU_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
GC10_MAP = {
    0: 6,  # punching_hole
    1: 7,  # welding_line
    2: 8,  # crescent_gap
    3: 9,  # water_spot
    4: 10, # oil_spot
    5: 11, # silk_spot
    6: 1,  # inclusion (maps to NEU's inclusion)
    7: 12, # rolled_pit
    8: 13, # crease
    9: 14  # waist_folding
}

def merge_datasets(neu_dir, gc10_dir, output_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
        
        # 1. Copy NEU-DET
        neu_img_dir = os.path.join(neu_dir, "images", split)
        neu_lbl_dir = os.path.join(neu_dir, "labels", split)
        if os.path.exists(neu_img_dir):
            for f in os.listdir(neu_img_dir):
                if f.endswith(".jpg"):
                    shutil.copy2(os.path.join(neu_img_dir, f), os.path.join(output_dir, "images", split, f"neu_{f}"))
                    lbl = f.replace(".jpg", ".txt")
                    if os.path.exists(os.path.join(neu_lbl_dir, lbl)):
                        with open(os.path.join(neu_lbl_dir, lbl), "r") as lf:
                            lines = lf.readlines()
                        with open(os.path.join(output_dir, "labels", split, f"neu_{lbl}"), "w") as lf:
                            for line in lines:
                                parts = line.strip().split()
                                if parts:
                                    parts[0] = str(NEU_MAP.get(int(parts[0]), int(parts[0])))
                                    lf.write(" ".join(parts) + "\n")

        # 2. Copy GC10-DET
        gc10_img_dir = os.path.join(gc10_dir, "images", split)
        gc10_lbl_dir = os.path.join(gc10_dir, "labels", split)
        if os.path.exists(gc10_img_dir):
            for f in os.listdir(gc10_img_dir):
                if f.endswith(".jpg"):
                    shutil.copy2(os.path.join(gc10_img_dir, f), os.path.join(output_dir, "images", split, f"gc10_{f}"))
                    lbl = f.replace(".jpg", ".txt")
                    if os.path.exists(os.path.join(gc10_lbl_dir, lbl)):
                        with open(os.path.join(gc10_lbl_dir, lbl), "r") as lf:
                            lines = lf.readlines()
                        with open(os.path.join(output_dir, "labels", split, f"gc10_{lbl}"), "w") as lf:
                            for line in lines:
                                parts = line.strip().split()
                                if parts:
                                    parts[0] = str(GC10_MAP.get(int(parts[0]), int(parts[0])))
                                    lf.write(" ".join(parts) + "\n")

    # 3. Write unified data.yaml
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump({
            "path": output_dir,
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(UNIFIED_CLASSES)}
        }, f)
    print(f"✅ Unified dataset created at {output_dir} with {len(UNIFIED_CLASSES)} classes.")

if __name__ == "__main__":
    neu_dir = "/content/drive/MyDrive/AegisVision/data/processed/neu_det_yolo"
    gc10_dir = "/content/drive/MyDrive/AegisVision/data/processed/gc10_det_yolo"
    out_dir = "/content/drive/MyDrive/AegisVision/data/processed/unified_yolo"
    
    if os.path.exists(neu_dir) and os.path.exists(gc10_dir):
        merge_datasets(neu_dir, gc10_dir, out_dir)
    else:
        print("❌ Missing source datasets. Ensure both NEU and GC10 are processed.")
