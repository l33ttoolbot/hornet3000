#!/usr/bin/env python3
"""
Build Unified Hornet Detection Dataset

Combines:
1. Kaggle hornet3000 dataset (Asian Hornet, European Hornet, Wasp)
2. iNaturalist images auto-labeled with YOLO-World (Honey Bee)

Output: Unified YOLO-format dataset with consistent class names
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
import argparse


# Unified class mapping
CLASSES = {
    "asian_hornet": 0,      # Vespa velutina
    "european_hornet": 1,   # Vespa crabro
    "wasp": 2,              # Vespula vulgaris
    "honey_bee": 3,         # Apis mellifera
}

# Kaggle dataset class mapping (from original hornet3000)
KAGGLE_CLASSES = {
    "Vespa_velutina": 0,    # -> asian_hornet
    "Vespa_crabro": 1,      # -> european_hornet
    "Vespulla_vulgaris": 2, # -> wasp (note: typo in original dataset)
    "vespulla_vulgaris": 2, # alternative spelling
}

# Source directories
KAGGLE_DATA_DIR = Path(__file__).parent.parent.parent / "hornet-data-kaggle"
INATURALIST_DIR = Path(__file__).parent.parent.parent / "hornet-data-raw" / "inaturalist"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "hornet_unified"


def convert_kaggle_labels(label_dir: Path, output_label_dir: Path, class_mapping: Dict[str, int]) -> int:
    """Convert Kaggle labels to unified format."""
    converted = 0

    for label_file in label_dir.glob("*.txt"):
        # Read original labels
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Convert class IDs
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class = int(parts[0])
                # Kaggle classes: 0=velutina, 1=crabro, 2=vulgaris
                # Unified: 0=asian_hornet, 1=european_hornet, 2=wasp
                # Same order, no conversion needed!
                new_class = old_class  # Direct mapping
                parts[0] = str(new_class)
                new_lines.append(" ".join(parts) + "\n")
                converted += 1

        # Write converted labels
        output_file = output_label_dir / label_file.name
        with open(output_file, "w") as f:
            f.writelines(new_lines)

    return converted


def process_kaggle_dataset(kaggle_dir: Path, output_dir: Path, splits: List[str] = ["train", "val", "test"]):
    """Process Kaggle hornet3000 dataset."""

    print("\n" + "=" * 60)
    print("Processing Kaggle hornet3000 dataset")
    print("=" * 60)

    if not kaggle_dir.exists():
        print(f"Kaggle dataset not found at {kaggle_dir}")
        print("Please download from: https://www.kaggle.com/datasets/marcoryvandijk/vespa-velutina-v-crabro-vespulina-vulgaris")
        return 0

    total_images = 0

    for split in splits:
        images_src = kaggle_dir / split / "images"
        labels_src = kaggle_dir / split / "labels"

        if not images_src.exists():
            print(f"Split {split} not found, skipping...")
            continue

        # Create output directories
        images_dst = output_dir / split / "images"
        labels_dst = output_dir / split / "labels"
        images_dst.mkdir(parents=True, exist_ok=True)
        labels_dst.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_file in images_src.glob("*.jpg"):
            shutil.copy(img_file, images_dst / img_file.name)
            total_images += 1

        # Convert and copy labels
        if labels_src.exists():
            converted = convert_kaggle_labels(labels_src, labels_dst, KAGGLE_CLASSES)
            print(f"  {split}: {total_images} images, {converted} labels")

    return total_images


def process_inaturalist_with_yolo_world(inaturalist_dir: Path, output_dir: Path) -> int:
    """
    Auto-label iNaturalist images with YOLO-World.
    This requires ultralytics to be installed.
    """

    print("\n" + "=" * 60)
    print("Auto-labeling iNaturalist images with YOLO-World")
    print("=" * 60)

    try:
        from ultralytics import YOLOWorld
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return 0

    # Load YOLO-World model
    print("Loading YOLO-World model...")
    model = YOLOWorld("yolov8s-world.pt")

    # Set classes for detection
    model.set_classes(["asian hornet", "european hornet", "wasp", "honey bee"])

    # Process each species
    species_mapping = {
        "vespa_velutina": "asian_hornet",
        "vespa_crabro": "european_hornet",
        "vespula_vulgaris": "wasp",
        "apis_mellifera": "honey_bee",
    }

    output_images = output_dir / "inaturalist" / "images"
    output_labels = output_dir / "inaturalist" / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    total_labeled = 0

    for species_dir, class_name in species_mapping.items():
        species_path = inaturalist_dir / species_dir / "images"

        if not species_path.exists():
            print(f"  {species_dir}: not found, skipping...")
            continue

        # Get all images
        images = list(species_path.glob("*.jpg"))
        print(f"  {species_dir}: {len(images)} images")

        for img_path in images:
            # Run detection
            results = model.predict(str(img_path), verbose=False)

            # Get class ID
            class_id = CLASSES[class_name]

            # Get image dimensions for normalization
            from PIL import Image
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # Write label file
            label_file = output_labels / f"{img_path.stem}.txt"

            with open(label_file, "w") as f:
                for result in results:
                    for box in result.boxes:
                        # Get bounding box (normalized)
                        x_center = (box.xyxyn[0][0] + box.xyxyn[0][2]) / 2
                        y_center = (box.xyxyn[0][1] + box.xyxyn[0][3]) / 2
                        width = box.xyxyn[0][2] - box.xyxyn[0][0]
                        height = box.xyxyn[0][3] - box.xyxyn[0][1]

                        # Write label (use species class, not detected class)
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        total_labeled += 1

            # Copy image
            shutil.copy(img_path, output_images / img_path.name)

    print(f"\nTotal labels created: {total_labeled}")
    return total_labeled


def create_unified_dataset_yaml(output_dir: Path) -> None:
    """Create YOLO data.yaml for unified dataset."""

    yaml_content = f"""# Unified Hornet Detection Dataset
# Generated by build_unified_dataset.py

path: {output_dir.absolute()}

train: train/images
val: val/images
test: test/images

# Classes
nc: 4
names:
  0: asian_hornet
  1: european_hornet
  2: wasp
  3: honey_bee

# Class descriptions
# 0: Asian Hornet (Vespa velutina) - Invasive species, target
# 1: European Hornet (Vespa crabro) - Native, do not harm
# 2: Common Wasp (Vespula vulgaris) - Distractor
# 3: Honey Bee (Apis mellifera) - Beneficial, protect
"""

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nCreated {yaml_path}")


def merge_inaturalist_to_splits(output_dir: Path, splits: tuple = (0.7, 0.2, 0.1)):
    """Split iNaturalist data into train/val/test."""
    import random

    inat_images = output_dir / "inaturalist" / "images"
    inat_labels = output_dir / "inaturalist" / "labels"

    if not inat_images.exists():
        return

    print("\n" + "=" * 60)
    print("Splitting iNaturalist data into train/val/test")
    print("=" * 60)

    # Get all images
    images = list(inat_images.glob("*.jpg"))
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * splits[0])
    n_val = int(n_total * splits[1])

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images)
    ]:
        # Create directories
        img_dst = output_dir / split_name / "images"
        lbl_dst = output_dir / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        # Move files
        for img_path in split_images:
            # Move image
            shutil.move(str(img_path), str(img_dst / img_path.name))

            # Move label
            label_path = inat_labels / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.move(str(label_path), str(lbl_dst / label_path.name))

        print(f"  {split_name}: {len(split_images)} images")

    # Remove empty inaturalist directory
    shutil.rmtree(output_dir / "inaturalist", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Build unified hornet detection dataset")
    parser.add_argument("--kaggle", type=str, default=None, help="Path to Kaggle dataset")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO-World auto-labeling")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Set paths
    kaggle_dir = Path(args.kaggle) if args.kaggle else KAGGLE_DATA_DIR
    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    print("=" * 60)
    print("Unified Hornet Detection Dataset Builder")
    print("=" * 60)
    print(f"Kaggle data: {kaggle_dir}")
    print(f"iNaturalist data: {INATURALIST_DIR}")
    print(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Process Kaggle dataset
    kaggle_count = process_kaggle_dataset(kaggle_dir, output_dir)

    # Step 2: Auto-label iNaturalist with YOLO-World
    if not args.skip_yolo:
        inat_count = process_inaturalist_with_yolo_world(INATURALIST_DIR, output_dir)

        # Step 3: Split iNaturalist data
        merge_inaturalist_to_splits(output_dir)

    # Step 4: Create data.yaml
    create_unified_dataset_yaml(output_dir)

    print("\n" + "=" * 60)
    print("Dataset creation complete!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print("\nClass mapping:")
    for name, idx in sorted(CLASSES.items(), key=lambda x: x[1]):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()