#!/usr/bin/env python3
"""
LUBW Portal Downloader

Downloads images from the LUBW/Convotis Asian Hornet reporting portal.
URL Pattern: https://gmp.convotis.com/documents/d/global/anhang_YYYY-MM-NNNN-[ext]
"""

import os
import re
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import argparse


class LUBWDownloader:
    """Download images from LUBW portal."""

    BASE_URL = "https://gmp.convotis.com/documents/d/global/anhang_{}"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def try_download(self, date_str: str, number: int, extension: str = "jpg") -> Optional[Path]:
        """Try to download a specific document."""
        # Format: anhang_2026-02-0087-jpg
        doc_id = f"anhang_{date_str}-{number:04d}-{extension}"
        url = self.BASE_URL.format(doc_id)

        try:
            response = self.session.get(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                # Determine extension from content-type
                content_type = response.headers.get("content-type", "")
                if "jpeg" in content_type or "jpg" in content_type:
                    ext = "jpg"
                elif "png" in content_type:
                    ext = "png"
                elif "webp" in content_type:
                    ext = "webp"
                else:
                    ext = extension

                # Save file
                filename = f"{date_str}_{number:04d}.{ext}"
                output_path = self.output_dir / filename

                with open(output_path, "wb") as f:
                    f.write(response.content)

                return output_path
        except Exception as e:
            pass

        return None

    def scan_date(self, date: datetime, max_number: int = 500) -> int:
        """Scan all documents for a specific date."""
        date_str = date.strftime("%Y-%m")
        date_dir = self.output_dir / date.strftime("%Y-%m")
        date_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0

        for number in range(1, max_number + 1):
            for ext in ["jpg", "jpeg", "png", "webp"]:
                result = self.try_download(f"{date_str}", number, ext)
                if result:
                    print(f"  ✓ {result.name}")
                    downloaded += 1
                    break  # Found, don't try other extensions

            # Rate limiting
            time.sleep(0.1)

        return downloaded

    def scan_range(self, start_date: datetime, end_date: datetime) -> int:
        """Scan a range of months."""
        total = 0
        current = start_date

        while current <= end_date:
            print(f"\nScanning {current.strftime('%Y-%m')}...")
            count = self.scan_date(current)
            total += count
            print(f"  Downloaded: {count} images")

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return total


def main():
    parser = argparse.ArgumentParser(description="Download images from LUBW portal")
    parser.add_argument("--start", type=str, default="2025-09", help="Start date (YYYY-MM)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m")
    end_date = datetime.strptime(args.end, "%Y-%m") if args.end else datetime.now()

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent.parent / "hornet-data-raw" / "lubw" / "vespa_velutina"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="* 60)
    print("LUBW Portal Downloader")
    print("=" * 60)
    print(f"Date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    print(f"Output: {output_dir}")
    print()

    downloader = LUBWDownloader(output_dir)
    total = downloader.scan_range(start_date, end_date)

    print("\n" + "=" * 60)
    print(f"Complete! Downloaded {total} images")
    print("=" * 60)


if __name__ == "__main__":
    main()