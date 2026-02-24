# Hornet3000 Scripts

Data processing and training scripts for the Asian Hornet detection project.

## Scripts

### Data Download

#### `download_inaturalist.py`
Download images from iNaturalist API.

```bash
python scripts/download_inaturalist.py
```

**Features:**
- No API key required
- Downloads all 4 species (Vespa velutina, Vespa crabro, Vespula vulgaris, Apis mellifera)
- Research-grade observations only
- License tracking (CC0, CC-BY, CC-BY-SA)
- Default: 1000 images per species

#### `download_lubw.py`
Download images from LUBW/Convotis reporting portal.

```bash
python scripts/download_lubw.py --start 2025-09 --end 2026-02
```

### Preprocessing

#### `preprocess.py`
Process raw images into YOLO-format dataset.

```bash
# Default: 1280x720 (for Hailo-10H / 720p camera)
python scripts/preprocess.py

# Alternative: 640x640 (standard YOLO)
python scripts/preprocess.py --size 640x640

# Custom splits
python scripts/preprocess.py --splits 0.8,0.15,0.05
```

**Features:**
- Resize to target size
- Remove duplicates (MD5)
- Split into train/val/test
- Generate YOLO data.yaml

#### `deduplicate.py`
Find and remove duplicate images.

```bash
# Report only
python scripts/deduplicate.py

# Actually remove
python scripts/deduplicate.py --remove

# Custom directory
python scripts/deduplicate.py --dir /path/to/images
```

## Output Structure

```
hornet-data-raw/           # Raw downloaded images
├── inaturalist/
│   ├── vespa_velutina/
│   ├── vespa_crabro/
│   ├── vespula_vulgaris/
│   └── apis_mellifera/
├── lubw/
│   └── vespa_velutina/
│       ├── 2025-09/
│       ├── 2025-10/
│       └── ...
└── metadata/

hornet-data-processed/     # Processed images
├── vespa_velutina/
├── vespa_crabro/
├── vespula_vulgaris/
└── apis_mellifera/

datasets/                  # YOLO-format datasets
└── hornet_detection/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## Classes

| ID | Species | Common Name |
|----|---------|-------------|
| 0 | Vespa velutina | Asian Hornet |
| 1 | Vespa crabro | European Hornet |
| 2 | Vespula vulgaris | Common Wasp |
| 3 | Apis mellifera | Western Honey Bee |

## Hardware Configuration

- **Raspberry Pi 5** + **Hailo-10H AI Hat**
- **Logitech 720p Camera**
- **Recommended image size:** 1280×720 (native camera resolution)