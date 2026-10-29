"""
Step 1: Data Preparation
========================
Organizes the raw dataset into a unified folder structure suitable for
ImageDataGenerator (train/val/test split with class subfolders).

Classes:
  0 - Calculus
  1 - Caries
  2 - Gingivitis
  3 - Hypodontia
  4 - Mouth Ulcer
  5 - Tooth Discoloration
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path
from collections import Counter

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\adhem\Downloads\Oral Disease")
RAW_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

random.seed(SEED)
np.random.seed(SEED)

# ── Source mappings: class_name -> list of source directories ──────────────────
CLASS_SOURCES = {
    "Calculus": [RAW_DIR / "Calculus" / "Calculus"],
    "Caries": [
        RAW_DIR / "Data caries" / "Data caries" / "caries augmented data set" / "preview",
    ],
    "Gingivitis": [RAW_DIR / "Gingivitis" / "Gingivitis"],
    "Hypodontia": [RAW_DIR / "hypodontia" / "hypodontia"],
    "Mouth_Ulcer": [
        RAW_DIR / "Mouth Ulcer" / "Mouth Ulcer" / "Mouth_Ulcer_augmented_DataSet" / "preview",
    ],
    "Tooth_Discoloration": [
        RAW_DIR / "Tooth Discoloration" / "images",
    ],
}


def collect_images(source_dirs):
    """Collect all image file paths from source directories."""
    images = []
    for src in source_dirs:
        if not src.exists():
            print(f"  ⚠ Source not found: {src}")
            continue
        for f in src.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)
    return images


def split_data(images, train_r, val_r, test_r):
    """Split image list into train, val, test sets."""
    random.shuffle(images)
    n = len(images)
    train_end = int(n * train_r)
    val_end = train_end + int(n * val_r)
    return images[:train_end], images[train_end:val_end], images[val_end:]


def main():
    print("=" * 60)
    print("  Oral Disease Dataset Preparation")
    print("=" * 60)

    # Clean existing dataset directory
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    summary = {}

    for class_name, sources in CLASS_SOURCES.items():
        print(f"\n📂 Processing class: {class_name}")
        images = collect_images(sources)
        print(f"   Found {len(images)} images")

        if len(images) == 0:
            print(f"   ⚠ No images found, skipping!")
            continue

        train_imgs, val_imgs, test_imgs = split_data(
            images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )

        summary[class_name] = {
            "total": len(images),
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            dest_dir = DATASET_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, img_path in enumerate(split_imgs):
                ext = img_path.suffix
                dest_path = dest_dir / f"{class_name}_{i:04d}{ext}"
                shutil.copy2(img_path, dest_path)

        print(
            f"   ✅ Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}"
        )

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"{'Class':<22} {'Total':>7} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("-" * 60)
    total_all = {"total": 0, "train": 0, "val": 0, "test": 0}
    for cls, counts in summary.items():
        print(
            f"{cls:<22} {counts['total']:>7} {counts['train']:>7} "
            f"{counts['val']:>7} {counts['test']:>7}"
        )
        for k in total_all:
            total_all[k] += counts[k]
    print("-" * 60)
    print(
        f"{'TOTAL':<22} {total_all['total']:>7} {total_all['train']:>7} "
        f"{total_all['val']:>7} {total_all['test']:>7}"
    )
    print("=" * 60)
    print(f"\n✅ Dataset saved to: {DATASET_DIR}")


if __name__ == "__main__":
    main()
