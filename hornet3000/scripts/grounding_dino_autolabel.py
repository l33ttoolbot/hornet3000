#!/usr/bin/env python3
"""
Grounding DINO Auto-Labeler for Hornet3000

Zero-Shot Object Detection für Hornissen, Wespen und Bienen.
Läuft CPU-only (Intel 14000 empfohlen) oder GPU.

Usage:
    python grounding_dino_autolabel.py --input ../hornet-data-raw/inaturalist --output ../hornet-data-raw/inaturalist_labels

Requirements:
    pip install torch torchvision transformers accelerate pillow tqdm
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "IDEA-Research/grounding-dino-tiny"  # Faster, good for CPU
# MODEL_ID = "IDEA-Research/grounding-dino-base"  # More accurate, slower

TEXT_PROMPT = "hornet. wasp. bee. insect."

BOX_THRESHOLD = 0.35   # Lower = more detections (but more false positives)
TEXT_THRESHOLD = 0.30  # Lower = more text matches

# Class mapping: detected label -> YOLO class ID
# Will be overridden by subfolder name if available
CLASS_MAP = {
    "hornet": 0,      # asian_hornet (default, needs manual review)
    "wasp": 2,        # wasp
    "bee": 3,         # honey_bee
    "insect": 0,      # default to hornet, needs review
}

# Species folder -> YOLO class mapping
SPECIES_MAP = {
    "vespa_velutina": 0,    # asian_hornet
    "vespa_crabro": 1,      # european_hornet
    "vespula_vulgaris": 2,  # wasp
    "apis_mellifera": 3,    # honey_bee
}


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_id=MODEL_ID, device="auto"):
    """
    Load Grounding DINO model.
    
    Args:
        model_id: HuggingFace model ID
        device: "auto", "cuda", "cpu", or "mps" (Apple Silicon)
    
    Returns:
        (processor, model, device)
    """
    print(f"\n{'='*60}")
    print(f"Loading Grounding DINO: {model_id}")
    print(f"{'='*60}")
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = "cpu"
    
    print(f"Device: {device}")
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params/1e6:.1f}M")
    
    return processor, model, device


# ============================================================================
# Detection
# ============================================================================

def detect_objects(image_path, processor, model, device, text_prompt=TEXT_PROMPT):
    """
    Run detection on a single image.
    
    Returns:
        dict with 'boxes', 'labels', 'scores' or None on error
    """
    try:
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        
        inputs = processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(img_h, img_w)]
        )
        
        result = results[0]
        
        # Convert to lists
        return {
            "boxes": [b.tolist() for b in result["boxes"]],
            "labels": result["labels"],
            "scores": [s.item() for s in result["scores"]],
            "width": img_w,
            "height": img_h,
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def to_yolo_format(box, img_w, img_h, class_id):
    """
    Convert bounding box to YOLO format.
    
    Args:
        box: [x1, y1, x2, y2] pixel coordinates
        img_w, img_h: image dimensions
        class_id: YOLO class ID
    
    Returns:
        "class_id cx cy w h" string (normalized)
    """
    x1, y1, x2, y2 = box
    
    # Clip to image bounds
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))
    
    # Convert to center format (normalized)
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    
    # Clamp values
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))
    
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ============================================================================
# Processing
# ============================================================================

def get_species_from_path(image_path, input_root):
    """
    Try to determine species from folder structure.
    
    Expected structure:
        input_root/species_name/images/xxx.jpg
    """
    rel_path = Path(image_path).relative_to(input_root)
    parts = rel_path.parts
    
    if len(parts) >= 1:
        folder = parts[0].lower()
        for species, class_id in SPECIES_MAP.items():
            if species in folder:
                return class_id
    
    return None


def process_folder(input_dir, output_dir, processor, model, device):
    """
    Process all images in a folder.
    
    Args:
        input_dir: Root folder with images
        output_dir: Output folder for labels
        processor, model, device: Grounding DINO components
    
    Returns:
        (processed_count, labelled_count, error_count)
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        images.extend(input_path.rglob(ext))
    
    print(f"\nFound {len(images)} images")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Statistics
    processed = 0
    labelled = 0
    errors = 0
    no_detection = 0
    species_stats = {}
    
    # Process each image
    for img_path in tqdm(images, desc="Labeling"):
        try:
            result = detect_objects(img_path, processor, model, device)
            
            if result is None:
                errors += 1
                continue
            
            if len(result["boxes"]) == 0:
                no_detection += 1
                processed += 1
                continue
            
            # Determine class ID
            # Priority: folder name > detected label
            folder_class = get_species_from_path(img_path, input_path)
            
            # Create label file (mirrors folder structure)
            rel_path = img_path.relative_to(input_path)
            label_path = output_path / (rel_path.with_suffix(".txt"))
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(label_path, "w") as f:
                for box, label, score in zip(
                    result["boxes"],
                    result["labels"],
                    result["scores"]
                ):
                    # Get class ID
                    if folder_class is not None:
                        class_id = folder_class
                    else:
                        # Fall back to detected label
                        label_lower = label.lower()
                        class_id = CLASS_MAP.get(label_lower, 0)
                    
                    # Write YOLO format
                    yolo_line = to_yolo_format(
                        box, result["width"], result["height"], class_id
                    )
                    f.write(yolo_line + "\n")
            
            labelled += 1
            processed += 1
            
            # Track species stats
            if folder_class is not None:
                species_name = [k for k, v in SPECIES_MAP.items() if v == folder_class][0]
            else:
                species_name = label.lower() if len(result["labels"]) > 0 else "unknown"
            
            species_stats[species_name] = species_stats.get(species_name, 0) + 1
            
        except Exception as e:
            print(f"\nError: {img_path}: {e}")
            errors += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Processed:      {processed}")
    print(f"Labeled:        {labelled}")
    print(f"No detections:  {no_detection}")
    print(f"Errors:         {errors}")
    print(f"\nBy species:")
    for species, count in sorted(species_stats.items()):
        print(f"  {species}: {count}")
    
    return processed, labelled, errors


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Auto-label images with Grounding DINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Label all images in a folder
    python grounding_dino_autolabel.py --input ./images --output ./labels
    
    # Use more accurate (but slower) model
    python grounding_dino_autolabel.py --input ./images --output ./labels --model IDEA-Research/grounding-dino-base
    
    # Force CPU
    python grounding_dino_autolabel.py --input ./images --output ./labels --device cpu
        """
    )
    
    parser.add_argument("--input", "-i", required=True,
                        help="Input folder with images")
    parser.add_argument("--output", "-o", required=True,
                        help="Output folder for labels")
    parser.add_argument("--model", "-m", default=MODEL_ID,
                        help=f"Model ID (default: {MODEL_ID})")
    parser.add_argument("--device", "-d", default="auto",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Device to use (default: auto)")
    parser.add_argument("--box-threshold", type=float, default=BOX_THRESHOLD,
                        help=f"Box confidence threshold (default: {BOX_THRESHOLD})")
    parser.add_argument("--text-threshold", type=float, default=TEXT_THRESHOLD,
                        help=f"Text confidence threshold (default: {TEXT_THRESHOLD})")
    
    args = parser.parse_args()
    
    # Update global thresholds
    global BOX_THRESHOLD, TEXT_THRESHOLD
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    
    # Load model
    processor, model, device = load_model(args.model, args.device)
    
    # Process
    start_time = datetime.now()
    process_folder(args.input, args.output, processor, model, device)
    elapsed = datetime.now() - start_time
    
    print(f"\nTime elapsed: {elapsed}")


if __name__ == "__main__":
    main()