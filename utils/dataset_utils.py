# ============================================================
#  utils/dataset_utils.py
#  Helpers for dataset preparation and augmentation checks
# ============================================================

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path


def count_dataset(dataset_path: str):
    """Print class distribution of a dataset folder."""
    print(f"\n{'='*45}")
    print(f"  Dataset : {dataset_path}")
    print(f"{'='*45}")
    total = 0
    for cls in sorted(os.listdir(dataset_path)):
        cls_path = os.path.join(dataset_path, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"  {cls:<20} : {n:>5} images")
            total += n
    print(f"  {'TOTAL':<20} : {total:>5} images")
    print(f"{'='*45}\n")


def verify_images(dataset_path: str):
    """Remove corrupted/unreadable images from dataset folders."""
    removed = 0
    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(root, fname)
            img   = cv2.imread(fpath)
            if img is None:
                print(f"[WARN] Removing corrupted image: {fpath}")
                os.remove(fpath)
                removed += 1
    print(f"[INFO] Verification complete. Removed {removed} corrupted files.")


def resize_images(dataset_path: str, target_size=(224, 224)):
    """Resize all images in the dataset to target_size in-place."""
    count = 0
    for root, dirs, files in os.walk(dataset_path):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(root, fname)
            img   = cv2.imread(fpath)
            if img is None:
                continue
            resized = cv2.resize(img, target_size)
            cv2.imwrite(fpath, resized)
            count += 1
    print(f"[INFO] Resized {count} images to {target_size}.")


def create_sample_dataset(output_dir: str, n_per_class: int = 10):
    """Create a tiny synthetic dataset for quick smoke-tests."""
    classes = ["with_mask", "without_mask"]
    for cls in classes:
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            # Random noise image that simulates a face patch
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(cls_dir, f"{cls}_{i:04d}.jpg"), img)
    print(f"[INFO] Sample dataset created at '{output_dir}' "
          f"({n_per_class} images/class).")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--action",  choices=["count", "verify", "resize", "sample"],
                    default="count")
    args = vars(ap.parse_args())

    if args["action"] == "count":
        count_dataset(args["dataset"])
    elif args["action"] == "verify":
        verify_images(args["dataset"])
    elif args["action"] == "resize":
        resize_images(args["dataset"])
    elif args["action"] == "sample":
        create_sample_dataset(args["dataset"])
