#!/usr/bin/env python3
"""
iNaturalist Image Downloader for Hornet Detection Dataset

Downloads images for 4 insect classes:
- Vespa velutina (Asian Hornet)
- Vespa crabro (European Hornet)
- Vespula vulgaris (Common Wasp)
- Apis mellifera (Honey Bee)

No API key required!
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# Species configuration (verified taxon IDs from iNaturalist API)
SPECIES = {
    "vespa_velutina": {"taxon_id": 119019, "name": "Asian Hornet"},
    "vespa_crabro": {"taxon_id": 54327, "name": "European Hornet"},
    "vespula_vulgaris": {"taxon_id": 127777, "name": "Common Wasp"},
    "apis_mellifera": {"taxon_id": 47219, "name": "Western Honey Bee"},
}

# iNaturalist API endpoints
API_BASE = "https://api.inaturalist.org/v1"
OBSERVATIONS_URL = f"{API_BASE}/observations"

# Output configuration
OUTPUT_BASE = Path(__file__).parent.parent.parent / "hornet-data-raw" / "inaturalist"
METADATA_FILE = OUTPUT_BASE / "metadata" / "inaturalist_observations.json"

# Request configuration
PER_PAGE = 200  # Max per page
MAX_IMAGES_PER_SPECIES = 1000
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nd,cc-by-nc-sa,cc-by-nc-nd"
RATE_LIMIT_DELAY = 1.0  # seconds between requests


def create_output_dirs():
    """Create output directories for all species."""
    for species_key in SPECIES:
        (OUTPUT_BASE / species_key / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / "metadata").mkdir(parents=True, exist_ok=True)


def fetch_observations(taxon_id: int, page: int = 1) -> dict:
    """Fetch observations from iNaturalist API."""
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",  # Only research-grade observations
        "photos": "true",  # Must have photos
        "license": LICENSES,
        "per_page": PER_PAGE,
        "page": page,
        "order": "desc",
        "order_by": "created_at",
    }
    
    response = requests.get(OBSERVATIONS_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def download_image(url: str, output_path: Path) -> bool:
    """Download a single image."""
    try:
        # Get original size URL
        original_url = url.replace("medium.", "original.")
        
        response = requests.get(original_url, timeout=30, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        elif response.status_code == 403:
            # Fallback to medium if original not available
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
    return False


def download_species(species_key: str, taxon_id: int, max_images: int = MAX_IMAGES_PER_SPECIES):
    """Download images for a single species."""
    print(f"\n{'='*60}")
    print(f"Downloading: {SPECIES[species_key]['name']} ({species_key})")
    print(f"Taxon ID: {taxon_id}")
    print(f"Max images: {max_images}")
    print(f"{'='*60}")
    
    output_dir = OUTPUT_BASE / species_key / "images"
    metadata = []
    downloaded = 0
    page = 1
    
    while downloaded < max_images:
        print(f"\nFetching page {page}...")
        
        try:
            data = fetch_observations(taxon_id, page)
        except Exception as e:
            print(f"Error fetching observations: {e}")
            break
        
        results = data.get("results", [])
        if not results:
            print("No more observations.")
            break
        
        total_results = data.get("total_results", 0)
        print(f"Found {total_results} total observations, {len(results)} on this page")
        
        for obs in results:
            if downloaded >= max_images:
                break
            
            obs_id = obs.get("id")
            photos = obs.get("photos", [])
            
            if not photos:
                continue
            
            for photo_idx, photo in enumerate(photos):
                if downloaded >= max_images:
                    break
                
                photo_url = photo.get("url", "")
                if not photo_url:
                    continue
                
                # Generate filename
                extension = photo_url.split(".")[-1].split("?")[0]
                filename = f"{obs_id}_{photo_idx}.{extension}"
                output_path = output_dir / filename
                
                # Skip if already exists
                if output_path.exists():
                    print(f"  Skipping existing: {filename}")
                    continue
                
                # Download image
                print(f"  [{downloaded+1}/{max_images}] Downloading: {filename}")
                if download_image(photo_url, output_path):
                    downloaded += 1
                    
                    # Store metadata
                    metadata.append({
                        "observation_id": obs_id,
                        "photo_id": photo.get("id"),
                        "filename": filename,
                        "url": photo_url,
                        "license": photo.get("license_code"),
                        "attribution": photo.get("attribution"),
                        "observed_on": obs.get("observed_on"),
                        "location": obs.get("location"),
                        "user_login": obs.get("user", {}).get("login"),
                        "downloaded_at": datetime.now().isoformat(),
                    })
                    
                    # Small delay between downloads
                    time.sleep(0.1)
        
        page += 1
        time.sleep(RATE_LIMIT_DELAY)
    
    # Save metadata
    metadata_file = OUTPUT_BASE / "metadata" / f"{species_key}_observations.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDownloaded {downloaded} images for {species_key}")
    return downloaded


def main():
    """Main download function."""
    print("="*60)
    print("iNaturalist Image Downloader for Hornet Dataset")
    print("="*60)
    
    create_output_dirs()
    
    total_downloaded = 0
    for species_key, info in SPECIES.items():
        count = download_species(species_key, info["taxon_id"])
        total_downloaded += count
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Downloaded {total_downloaded} images total")
    print(f"Output: {OUTPUT_BASE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()