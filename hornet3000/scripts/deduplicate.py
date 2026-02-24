#!/usr/bin/env python3
"""
Deduplicate images in the hornet dataset.

Uses perceptual hashing and exact MD5 matching.
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import argparse


def get_md5_hash(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_exact_duplicates(base_dir: Path) -> dict:
    """Find exact duplicates by MD5 hash."""
    hashes = defaultdict(list)
    
    for img_path in base_dir.rglob("*.jpg"):
        img_hash = get_md5_hash(img_path)
        hashes[img_hash].append(img_path)
    
    # Filter to only duplicates
    return {h: paths for h, paths in hashes.items() if len(paths) > 1}


def find_duplicates_by_size(base_dir: Path) -> dict:
    """Find potential duplicates by file size."""
    sizes = defaultdict(list)
    
    for img_path in base_dir.rglob("*.jpg"):
        size = img_path.stat().st_size
        sizes[size].append(img_path)
    
    # Filter to only duplicates
    return {s: paths for s, paths in sizes.items() if len(paths) > 1}


def remove_duplicates(base_dir: Path, dry_run: bool = True) -> int:
    """Remove duplicate images, keeping the first one found."""
    duplicates = find_exact_duplicates(base_dir)
    
    removed_count = 0
    
    for img_hash, paths in duplicates.items():
        # Keep the first one, remove the rest
        to_remove = paths[1:]
        
        for path in to_remove:
            if dry_run:
                print(f"Would remove: {path}")
            else:
                print(f"Removing: {path}")
                path.unlink()
            removed_count += 1
    
    return removed_count


def main():
    parser = argparse.ArgumentParser(description="Deduplicate hornet dataset images")
    parser.add_argument("--dir", type=str, default=None, help="Directory to scan")
    parser.add_argument("--remove", action="store_true", help="Actually remove duplicates")
    parser.add_argument("--report", action="store_true", help="Only report, don't remove")
    args = parser.parse_args()
    
    # Default directory
    if args.dir:
        base_dir = Path(args.dir)
    else:
        base_dir = Path(__file__).parent.parent.parent / "hornet-data-raw"
    
    print(f"Scanning: {base_dir}")
    print()
    
    # Find duplicates
    print("Finding exact duplicates by MD5...")
    duplicates = find_exact_duplicates(base_dir)
    
    total_files = sum(1 for _ in base_dir.rglob("*.jpg"))
    duplicate_count = sum(len(paths) - 1 for paths in duplicates.values())
    
    print(f"Total images: {total_files}")
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Duplicates to remove: {duplicate_count}")
    
    if duplicates:
        print("\nDuplicate groups:")
        for img_hash, paths in list(duplicates.items())[:10]:  # Show first10
            print(f"\n  Hash: {img_hash[:16]}...")
            for p in paths:
                print(f"    - {p}")
        
        if len(duplicates) > 10:
            print(f"\n  ... and {len(duplicates) - 10} more groups")
    
    if args.remove and duplicate_count > 0:
        print("\nRemoving duplicates...")
        removed = remove_duplicates(base_dir, dry_run=False)
        print(f"Removed {removed} duplicate files")
    elif args.report or not args.remove:
        print("\nRun with --remove to actually delete duplicates")


if __name__ == "__main__":
    main()