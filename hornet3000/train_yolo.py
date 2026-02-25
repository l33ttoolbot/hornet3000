#!/usr/bin/env python3
"""
Hornet3000 YOLO Training Script
Trains a YOLO model on the hornet dataset for later auto-labeling.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO for hornet detection")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (yolov8n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default=None, help="Device (cuda, cpu, mps, or auto)")
    args = parser.parse_args()

    dataset_path = Path(__file__).parent.parent / "hornet-yolo" / "dataset.yaml"
    
    print(f"Dataset: {dataset_path}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Image size: {args.imgsz}")
    
    # Load model
    model = YOLO(args.model)
    
    # Train
    results = model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="hornet3000",
        name="yolo_detect",
        exist_ok=True,
        verbose=True
    )
    
    print(f"\nTraining complete!")
    print(f"Best model: hornet3000/yolo_detect/weights/best.pt")
    
    # Validate
    metrics = model.val()
    print(f"\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()