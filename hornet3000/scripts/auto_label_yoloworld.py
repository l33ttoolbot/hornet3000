#!/usr/bin/env python3
"""
Auto-label images using YOLO-World.

No training required - zero-shot detection for hornets and bees!
"""

from pathlib import Path
import argparse


CLASSES = {
    "asian_hornet": 0,      # Vespa velutina
    "european_hornet": 1,   # Vespa crabro
    "wasp": 2,              # Vespula vulgaris
    "honey_bee": 3,         # Apis mellifera
}

# YOLO-World prompt mapping
YW_PROMPTS = {
    "vespa_velutina": "asian hornet",
    "vespa_crabro": "european hornet",
    "vespula_vulgaris": "wasp",
    "apis_mellifera": "honey bee",
}


def auto_label_with_yolo_world(
    input_dir: Path,
    output_dir: Path,
    species: str = None,
    confidence: float = 0.25
):
    """
    Auto-label images using YOLO-World zero-shot detection.

    Args:
        input_dir: Directory with images (species subdirectories)
        output_dir: Output directory for labels
        species: Specific species to process (None = all)
        confidence: Minimum confidence threshold
    """
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        print("ERROR: ultralytics not installed!")
        print("Run: pip install ultralytics")
        return 0

    print("Loading YOLO-World model (yolov8s-world.pt)...")
    model = YOLOWorld("yolov8s-world.pt")

    # Set all class prompts
    model.set_classes(list(YW_PROMPTS.values()))

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    total_labeled = 0
    total_images = 0

    # Process each species directory
    for species_dir, prompt in YW_PROMPTS.items():
        if species and species_dir != species:
            continue

        species_path = input_dir / species_dir
        if not species_path.exists():
            print(f"  {species_dir}: directory not found, skipping...")
            continue

        # Find images
        images = list(species_path.rglob("*.jpg")) + list(species_path.rglob("*.png"))

        if not images:
            print(f"  {species_dir}: no images found")
            continue

        print(f"\n{species_dir} ({prompt}):")
        print(f"  Found {len(images)} images")

        # Get class ID
        class_name = species_dir  # Maps directly
        class_id = CLASSES.get(class_name.replace("_images", "").replace("images", ""))
        if class_id is None:
            # Try direct mapping
            class_id = CLASSES.get(class_name)
        if class_id is None:
            print(f"  WARNING: Unknown class for {species_dir}, using 0")
            class_id = 0

        labeled = 0

        for i, img_path in enumerate(images):
            if (i + 1) % 50 == 0:
                print(f"  Processing {i+1}/{len(images)}...")

            try:
                # Run detection
                results = model.predict(
                    str(img_path),
                    conf=confidence,
                    verbose=False
                )

                # Get image dimensions
                from PIL import Image
                with Image.open(img_path) as img:
                    img_w, img_h = img.size

                # Write label file
                label_file = labels_dir / f"{img_path.stem}.txt"

                with open(label_file, "w") as f:
                    for result in results:
                        if result.boxes is None:
                            continue

                        for box in result.boxes:
                            # Filter by confidence
                            if box.conf[0] < confidence:
                                continue

                            # Get normalized bounding box
                            # xywhn format: x_center, y_center, width, height (normalized)
                            if hasattr(box, 'xywhn') and box.xywhn is not None:
                                xywhn = box.xywhn[0]
                                x_center, y_center, width, height = xywhn
                            else:
                                # Convert from xyxy
                                xyxy = box.xyxyn[0]
                                x_center = (xyxy[0] + xyxy[2]) / 2
                                y_center = (xyxy[1] + xyxy[3]) / 2
                                width = xyxy[2] - xyxy[0]
                                height = xyxy[3] - xyxy[1]

                            # Always use the species class (not detected class)
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            labeled += 1

                total_images += 1

            except Exception as e:
                print(f"  Error processing {img_path}: {e}")

        print(f"  Labeled: {labeled} boxes")
        total_labeled += labeled

    print(f"\n{'='*60}")
    print(f"Total: {total_images} images, {total_labeled} labels")
    print(f"Output: {labels_dir}")

    return total_labeled


def main():
    parser = argparse.ArgumentParser(description="Auto-label images with YOLO-World")
    parser.add_argument("--input", type=str, default=None, help="Input directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--species", type=str, default=None, help="Specific species to process")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    # Default paths
    input_dir = Path(args.input) if args.input else Path(__file__).parent.parent.parent / "hornet-data-raw" / "inaturalist"
    output_dir = Path(args.output) if args.output else Path(__file__).parent.parent.parent / "hornet-data-raw" / "inaturalist_labels"

    print("=" * 60)
    print("YOLO-World Auto-Labeling")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Confidence: {args.confidence}")
    print()

    auto_label_with_yolo_world(input_dir, output_dir, args.species, args.confidence)


if __name__ == "__main__":
    main()