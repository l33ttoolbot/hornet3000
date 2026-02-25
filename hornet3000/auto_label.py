#!/usr/bin/env python3
"""
Hornet3000 Auto-Labeling Script
Uses trained YOLO model to label images without annotations.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import shutil

def main():
    parser = argparse.ArgumentParser(description="Auto-label images with trained YOLO")
    parser.add_argument("--model", required=True, help="Path to trained model (best.pt)")
    parser.add_argument("--input", required=True, help="Input directory with images")
    parser.add_argument("--output", required=True, help="Output directory for YOLO format")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--copy-images", action="store_true", help="Copy images to output")
    args = parser.parse_args()

    model = YOLO(args.model)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in input_dir.rglob("*") if f.suffix.lower() in image_extensions]
    print(f"Found {len(images)} images")

    labeled_count = 0
    skipped_count = 0

    for img_path in images:
        # Run inference
        results = model(img_path, conf=args.conf, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Save labels
            label_name = img_path.stem + ".txt"
            label_path = labels_dir / label_name
            
            with open(label_path, "w") as f:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    x, y, w, h = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            # Optionally copy image
            if args.copy_images:
                shutil.copy(img_path, images_dir / img_path.name)
            
            labeled_count += 1
        else:
            skipped_count += 1
        
        if (labeled_count + skipped_count) % 100 == 0:
            print(f"Progress: {labeled_count + skipped_count}/{len(images)} (labeled: {labeled_count})")

    print(f"\nDone!")
    print(f"Labeled: {labeled_count}")
    print(f"Skipped (no detection): {skipped_count}")
    print(f"Output: {output_dir}")

if __name__ == "__main__":
    main()