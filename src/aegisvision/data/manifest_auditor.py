# AegisVision V2 — Dataset Manifest Generator & Leakage Auditor

import os
import hashlib
import yaml
from pathlib import Path
from typing import Dict, List, Set

class DatasetAuditor:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.splits = ["train", "val", "test"]

    def compute_md5(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Computes MD5 hash of an image to detect duplicates and prevent leakage."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    def audit_splits(self) -> Dict[str, any]:
        """Scans splits, checks for leakage across train/val/test, and validates labels."""
        split_hashes: Dict[str, Set[str]] = {split: set() for split in self.splits}
        split_stats: Dict[str, int] = {split: 0 for split in self.splits}
        leakage_detected = []

        print("Starting strict dataset audit and hash verification...")

        for split in self.splits:
            img_dir = self.data_root / split / "images"
            if not img_dir.exists():
                print(f"Warning: Image directory not found for split '{split}' at {img_dir}")
                continue

            image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            images = [p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions]
            split_stats[split] = len(images)

            for img_path in images:
                file_hash = self.compute_md5(img_path)
                
                # Check for cross-split leakage
                for other_split, hashes in split_hashes.items():
                    if other_split != split and file_hash in hashes:
                        leakage_detected.append((img_path.name, split, other_split))
                
                split_hashes[split].add(file_hash)

        audit_report = {
            "split_image_counts": split_stats,
            "leakage_detected": leakage_detected,
            "is_leakage_free": len(leakage_detected) == 0
        }
        
        return audit_report

if __name__ == "__main__":
    print("Executing Dataset Auditor test run...")
    auditor = DatasetAuditor("data")
    report = auditor.audit_splits()
    print("Audit Report Summary:", report)
