# Second PC Setup for Grounding DINO Auto-Labeling

## Übersicht

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATENFLUSS                                │
│                                                                  │
│  Raspberry Pi 5                          Intel 14000 PC         │
│  (192.168.178.52)                        (Dein Zweitrechner)    │
│                                                                  │
│  hornet-data-raw/                        hornet-data-raw/       │
│  ├── inaturalist/    ────── SSH ──────►  ├── inaturalist/       │
│  │   ├── vespa_velutina/                 │   ...                │
│  │   ├── vespa_crabro/                                          │
│  │   ├── vespula_vulgaris/                                      │
│  │   └── apis_mellifera/                                        │
│  └── lubw/                                                      │
│                                                                  │
│                    ◄───── SSH ──────                            │
│  hornet-data-raw/                        hornet-data-raw/       │
│  └── inaturalist_labels/ ◄────────────── └── inaturalist_labels/│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Schritt 1: Verbindung testen

### Vom Intel PC zum Raspberry Pi:

```bash
# IP des Raspberry Pi: 192.168.178.52
# User: tool

ssh tool@192.168.178.52

# Wenn klappt: Verbindung funktioniert!
exit
```

---

## Schritt 2: Intel PC Setup

### 2.1 Repository klonen

```bash
# Auf dem Intel PC
cd ~
git clone https://github.com/l33ttoolbot/hornet3000.git
cd hornet3000
```

### 2.2 Python Environment

```bash
# Python 3.10+ benötigt
python3 --version

# Virtual Environment erstellen
python3 -m venv venv
source venv/bin/activate

# Abhängigkeiten installieren (CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate pillow tqdm

# Testen
python -c "from transformers import AutoModelForZeroShotObjectDetection; print('OK')"
```

### 2.3 Daten vom Raspberry Pi kopieren

```bash
# Ordner erstellen
mkdir -p ../hornet-data-raw

# rsync ist effizienter als scp
# iNaturalist Daten kopieren (~5GB)
rsync -avz --progress tool@192.168.178.52:/home/tool/.openclaw/workspace-main/hornet-data-raw/inaturalist ../hornet-data-raw/

# Optional: LUBW Daten (~1GB)
rsync -avz --progress tool@192.168.178.52:/home/tool/.openclaw/workspace-main/hornet-data-raw/lubw ../hornet-data-raw/
```

---

## Schritt 3: Auto-Labeling ausführen

```bash
# Im hornet3000 Ordner
source venv/bin/activate

# Grounding DINO starten (läuft 3-4 Stunden)
python scripts/grounding_dino_autolabel.py \
    --input ../hornet-data-raw/inaturalist \
    --output ../hornet-data-raw/inaturalist_labels

# Oder mit Screen/Tmux im Hintergrund:
screen -S grounding
python scripts/grounding_dino_autolabel.py \
    --input ../hornet-data-raw/inaturalist \
    --output ../hornet-data-raw/inaturalist_labels
# Ctrl+A, D to detach
# screen -r grounding to reattach
```

---

## Schritt 4: Labels zurück zum Raspberry Pi

```bash
# Nach dem Labeling (auf Intel PC)
rsync -avz --progress ../hornet-data-raw/inaturalist_labels tool@192.168.178.52:/home/tool/.openclaw/workspace-main/hornet-data-raw/
```

---

## Alternative: NFS Share (permanent)

### Auf dem Raspberry Pi (Server):

```bash
# NFS Server installieren
sudo apt install nfs-kernel-server

# Freigabe erstellen
echo "/home/tool/.openclaw/workspace-main/hornet-data-raw *(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports

# NFS starten
sudo systemctl restart nfs-kernel-server

# Firewall (falls aktiv)
sudo ufw allow from 192.168.178.0/24 to any port nfs
```

### Auf dem Intel PC (Client):

```bash
# NFS Client installieren
sudo apt install nfs-common

# Mount point erstellen
sudo mkdir -p /mnt/hornet-data

# Mounten
sudo mount 192.168.178.52:/home/tool/.openclaw/workspace-main/hornet-data-raw /mnt/hornet-data

# Permanent (in /etc/fstab):
# 192.168.178.52:/home/tool/.openclaw/workspace-main/hornet-data-raw /mnt/hornet-data nfs defaults 0 0

# Jetzt direkt auf den Daten arbeiten:
python scripts/grounding_dino_autolabel.py \
    --input /mnt/hornet-data/inaturalist \
    --output /mnt/hornet-data/inaturalist_labels
```

---

## SSH Key für passwortlosen Login (optional)

```bash
# Auf dem Intel PC
ssh-keygen -t ed25519
ssh-copy-id tool@192.168.178.52

# Test
ssh tool@192.168.178.52 "hostname"
```

---

## Performance-Tipps

### Batch-Processing (mehrere Prozesse)

```bash
# Auf Intel PC mit vielen Kernen
# Terminal 1: Vespa velutina
python scripts/grounding_dino_autolabel.py \
    --input ../hornet-data-raw/inaturalist/vespa_velutina \
    --output ../hornet-data-raw/inaturalist_labels/vespa_velutina

# Terminal 2: Vespa crabro
python scripts/grounding_dino_autolabel.py \
    --input ../hornet-data-raw/inaturalist/vespa_crabro \
    --output ../hornet-data-raw/inaturalist_labels/vespa_crabro

# Terminal 3: Vespula vulgaris
python scripts/grounding_dino_autolabel.py \
    --input ../hornet-data-raw/inaturalist/vespula_vulgaris \
    --output ../hornet-data-raw/inaturalist_labels/vespula_vulgaris

# Terminal 4: Apis mellifera
python scripts/grounding_dino_autolabel.py \
    --input ../hornet-data-raw/inaturalist/apis_mellifera \
    --output ../hornet-data-raw/inaturalist_labels/apis_mellifera
```

### RAM-Nutzung optimieren

```bash
# Modell im RAM halten (bei 32GB kein Problem)
export TRANSFORMERS_OFFLINE=1  # Nach erstem Download
```

---

## Troubleshooting

### "CUDA not available"
→ Normal auf Intel PC, läuft auf CPU

### "Out of memory"
→ `grounding-dino-tiny` statt `grounding-dino-base` verwenden

### "Connection refused"
→ SSH Port 22 evtl. blockiert: `sudo ufw allow 22`

### Sehr langsam
→ Andere Programme schließen, nur 1 Prozess pro CPU

---

## Zeitschätzung

| Spezies | Bilder | Zeit (Intel 14000) |
|---------|--------|-------------------|
| Vespa velutina | ~2.100 | ~1 Stunde |
| Vespa crabro | ~1.200 | ~40 min |
| Vespula vulgaris | ~1.000 | ~30 min |
| Apis mellifera | ~1.000 | ~30 min |
| **Gesamt** | **~5.300** | **~3-4 Stunden** |