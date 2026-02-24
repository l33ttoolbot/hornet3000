#!/usr/bin/env python3
"""
Preprocessing Pipeline for Hornet Detection Dataset

- Resize images to target size (default: 1280x720 for Hailo-10H)
- Convert to unified format (JPG)
- Split into train/val/test
- Generate YOLO-format labels
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple
import random
import argparse
from datetime import datetime


# Configuration
DEFAULT_IMG_SIZE = (1280, 720)  # For Hailo-10H / Logitech 720p
ALT_IMG_SIZE = (640, 640)  # Standard YOLO

# Classes (YOLO format)
CLASSES = {
    "vespa_velutina": 0,      # Asian Hornet
    "vespa_crabro": 1,        # European Hornet
    "vespula_vulgaris": 2,    # Common Wasp
    "apis_mellifera": 3,      # Honey Bee
}

# Paths
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "hornet-data-raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "hornet-data-processed"
DATASET_DIR = Path(__file__).parent.parent.parent / "datasets"


def get_image_hash(img_path: Path) -> str:
    """Calculate MD5 hash of image file."""
    with open(img_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_duplicates(image_dir: Path) -> Dict[str, List[Path]]:
    """Find duplicate images by hash."""
    hashes = {}
    duplicates = {}

    for img_path in image_dir.rglob("*.jpg"):
        img_hash = get_image_hash(img_path)
        if img_hash in hashes:
            if img_hash not in duplicates:
                duplicates[img_hash] = [hashes[img_hash]]
            duplicates[img_hash].append(img_path)
        else:
            hashes[img_hash] = img_path

    return duplicates


def resize_image(img_path: Path, output_path: Path, size: Tuple[int, int]) -> bool:
    """Resize image to target size, maintaining aspect ratio."""
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resize maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Create new image with exact size (pad if necessary)
            new_img = Image.new("RGB", size, (0, 0, 0))
            offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
            new_img.paste(img, offset)

            new_img.save(output_path, "JPEG", quality=95)
            return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def process_raw_images(
    source_dirs: List[Path],
    output_dir: Path,
    img_size: Tuple[int, int],
    remove_duplicates: bool = True
) -> Dict[str, int]:
    """Process raw images from multiple sources."""

    stats = {"total": 0, "processed": 0, "duplicates_removed": 0, "errors": 0}
    all_hashes = {}

    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Source not found: {source_dir}")
            continue

        # Each subfolder is a class
        for class_dir in source_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_name not in CLASSES:
                print(f"Unknown class: {class_name}")
                continue

            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {class_name}...")

            for img_path in class_dir.rglob("*.jpg"):
                stats["total"] += 1

                # Check for duplicates
                if remove_duplicates:
                    img_hash = get_image_hash(img_path)
                    if img_hash in all_hashes:
                        stats["duplicates_removed"] += 1
                        continue
                    all_hashes[img_hash] = img_path

                # Process image
                output_path = output_class_dir / f"{img_path.stem}.jpg"
                if resize_image(img_path, output_path, img_size):
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1

    return stats


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1)
) -> None:
    """Split dataset into train/val/test."""

    train_ratio, val_ratio, test_ratio = splits

    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for class_name in source_dir.iterdir():
        if not class_name.is_dir():
            continue

        images = list(class_name.glob("*.jpg"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        print(f"\n{class_name.name}:")
        print(f"  Train: {len(train_images)}")
        print(f"  Val: {len(val_images)}")
        print(f"  Test: {len(test_images)}")

        for split_name, split_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images)
        ]:
            for img_path in split_images:
                # Copy image
                dst_img = output_dir / split_name / "images" / f"{class_name.name}_{img_path.name}"
                shutil.copy(img_path, dst_img)

                # Create empty label file (for detection without bboxes)
                # These need to be annotated later!
                label_path = output_dir / split_name / "labels" / f"{class_name.name}_{img_path.stem}.txt"
                class_id = CLASSES[class_name.name]
                # Placeholder: full-image bbox for detection
                # Format: class_id cx cy w h
                with open(label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def create_data_yaml(output_dir: Path, img_size: Tuple[int, int]) -> None:
    """Create YOLO data.yaml config."""

    yaml_content = f"""# Hornet Detection Dataset
path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: asian_hornet
  1: european_hornet
  2: wasp
  3: honey_bee

# Image size
imgsz: {img_size[0]}
"""

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nCreated {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess hornet detection dataset")
    parser.add_argument("--size", type=str, default="1280x720", help="Image size (WxH)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip duplicate removal")
    parser.add_argument("--splits", type=str, default="0.7,0.2,0.1", help="Train,val,test splits")
    args = parser.parse_args()

    # Parse arguments
    img_size = tuple(map(int, args.size.lower().split("x")))
    splits = tuple(map(float, args.splits.split(",")))
    remove_duplicates = not args.no_dedup

    print("=" * 60)
    print("Hornet Detection Dataset - Preprocessing Pipeline")
    print("=" * 60)
    print(f"Image size: {img_size}")
    print(f"Remove duplicates: {remove_duplicates}")
    print(f"Splits: {splits}")
    print()

    # Source directories
    source_dirs = [
        RAW_DATA_DIR / "inaturalist",
        RAW_DATA_DIR / "lubw",
        RAW_DATA_DIR / "flickr",
        RAW_DATA_DIR / "waarnemingen",
    ]

    # Step 1: Process raw images
    print("Step 1: Processing raw images...")
    processed_dir = PROCESSED_DATA_DIR / "processed"
    stats = process_raw_images(source_dirs, processed_dir, img_size, remove_duplicates)
    print(f"\nProcessed: {stats['processed']}/{stats['total']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Errors: {stats['errors']}")

    # Step 2: Split dataset
    print("\nStep 2: Splitting dataset...")
    dataset_dir = DATASET_DIR / "hornet_detection"
    split_dataset(processed_dir, dataset_dir, splits)

    # Step 3: Create data.yaml
    print("\nStep 3: Creating data.yaml...")
    create_data_yaml(dataset_dir, img_size)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output: {dataset_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()