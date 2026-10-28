"""AegisVision - Severstal RLE to YOLO-Seg Polygon Pipeline.
Converts RLE masks to pixel-perfect polygons, including background (clean) images.
"""
import os
import cv2
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def rle2mask(rle, shape=(256, 1600)):
    if not isinstance(rle, str) or rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)
    s = rle.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start+length] = 1
    return mask.reshape(shape, order="F")

def main():
    dataset_path = "/content/severstal_data"
    local_out = "/content/severstal_seg_local"
    
    # 1. Setup dirs
    for split in ['train', 'val']:
        os.makedirs(os.path.join(local_out, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(local_out, 'labels', split), exist_ok=True)
        
    # 2. Load Data
    df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
    grouped = df.groupby('ImageId')
    images_with_defects = set(df['ImageId'].unique())
    all_images = set(os.listdir(os.path.join(dataset_path, "train_images")))
    clean_images = list(all_images - images_with_defects)
    
    # 3. Split
    unique_ids = list(images_with_defects)
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    train_clean, val_clean = train_test_split(clean_images, test_size=0.2, random_state=42)
    
    # 4. Process Defect Images
    for split_name, ids in [('train', train_ids), ('val', val_ids)]:
        for img_id in ids:
            img_path = os.path.join(dataset_path, "train_images", img_id)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: continue
            h, w = image.shape
            
            group = grouped.get_group(img_id)
            polygon_lines = []
            
            for _, row in group.iterrows():
                class_id = int(row['ClassId']) - 1 
                rle = row['EncodedPixels']
                if pd.isna(rle): continue
                    
                mask = rle2mask(rle, shape=(h, w))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 100: continue
                    polygon = contour.reshape(-1, 2).astype(float)
                    polygon[:, 0] /= w
                    polygon[:, 1] /= h
                    poly_str = " ".join([f"{coord:.6f}" for coord in polygon.flatten()])
                    polygon_lines.append(f"{class_id} {poly_str}")
                    
            if polygon_lines:
                cv2.imwrite(os.path.join(local_out, "images", split_name, img_id), image)
                with open(os.path.join(local_out, "labels", split_name, os.path.splitext(img_id)[0] + ".txt"), "w") as f:
                    f.write("\n".join(polygon_lines))
                    
    # 5. Process Clean Images
    for split_name, ids in [('train', train_clean), ('val', val_clean)]:
        for img_id in ids:
            src = os.path.join(dataset_path, "train_images", img_id)
            dst = os.path.join(local_out, "images", split_name, img_id)
            shutil.copy2(src, dst)
            # Empty label file for clean images
            with open(os.path.join(local_out, "labels", split_name, os.path.splitext(img_id)[0] + ".txt"), "w") as f:
                f.write("")
                
    print("Dataset conversion complete.")

if __name__ == "__main__":
    main()
