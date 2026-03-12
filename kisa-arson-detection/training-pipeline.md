# Arson Detection — Training Pipeline

## Overview

End-to-end pipeline for training the D-FINE Nano arson detection model. Three training rounds were conducted with incremental data augmentation.

```
Round 1: Internet fire images → Domain mismatch failure
Round 2: + KISA CCTV frames + Roboflow CCTV data → 3-class success
Round 3: + Person data augmentation (intrusion + auto-label) → 10/10 PASS
```

## Dataset Evolution

| Round | Train | Val | Classes | Sources |
|-------|-------|-----|---------|---------|
| 1st | 8,695 | 1,208 | 2 (person, fire_smoke) | Internet fire images |
| 2nd | 21,901 | 2,516 | 3 (person, fire, smoke) | + KISA frames + Roboflow CCTV |
| 3rd | 21,901 | 2,516 | 3 (person, fire, smoke) | + 534 intrusion person + 440 auto-labeled person |

### Final Dataset Composition (Round 3)

| Class | BBox Count | Sources |
|-------|-----------|---------|
| person | 26,020+ | Original dataset + intrusion cross-use + YOLO auto-label |
| fire | 24,951 | Original + KISA manual labeling + Roboflow CCTV |
| smoke | 7,676 | Original + KISA manual labeling + Roboflow CCTV |

## Data Pipeline

### Step 1: Frame Extraction

Extract frames around fire events from KISA test videos for manual labeling:

```bash
python extract_arson_frames.py
```

- Window: -30s to +30s around GT fire event (smoke may appear before fire)
- Sampling: 2fps
- Output: 1,510 frames from 10 KISA videos

### Step 2: Manual Labeling (CVAT)

- Uploaded 1,510 frames to CVAT
- Labeled fire and smoke bounding boxes manually
- Result: 770 labeled images, 732 fire bbox, 181 smoke bbox
- Export: YOLO 1.1 format

### Step 3: Dataset Merging

Merge multiple data sources into a unified 3-class dataset:

```bash
python merge_arson_dataset.py
```

**Class remapping** (critical step):
- Original dataset: 0=person, 1=fire, 2=smoke (already correct)
- CVAT export: 0=fire, 1=smoke → remapped to 1=fire, 2=smoke
- Roboflow data: 0=fire, 1=smoke → remapped to 1=fire, 2=smoke

### Step 4: Person Data Augmentation (Round 3)

The 2nd-round model had weak person detection. Two augmentation strategies:

1. **YOLO auto-labeling**: Ran YOLO11s on 1,510 KISA frames → 440 images with person bbox → CVAT manual review (7 images had false positives, removed)

2. **Intrusion cross-use**: Copied 534 manually-labeled person images from the intrusion detection dataset (already verified in CVAT)

```bash
python autolabel_person_arson.py   # YOLO auto-label
python add_intrusion_person_to_arson.py  # Cross-use
```

### Step 5: YOLO → COCO JSON Conversion

D-FINE requires COCO-format annotations:

```bash
python convert_yolo_to_coco.py \
    --src /path/to/arson_3class \
    --dst /path/to/arson_3class_coco \
    --classes person fire smoke
```

**YOLO format** (per line): `class_id cx cy w h` (normalized 0-1)

**COCO format** (JSON):
```json
{
  "images": [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 640}],
  "annotations": [{"id": 0, "image_id": 0, "category_id": 1,
                    "bbox": [x, y, w, h], "area": 1234.5, "iscrowd": 0}],
  "categories": [{"id": 0, "name": "person"}, {"id": 1, "name": "fire"}, {"id": 2, "name": "smoke"}]
}
```

### Step 6: Server Transfer

```bash
scp -r arson_3class_coco gmission@192.168.1.34:~/datasets/arson_3class_coco
scp configs/dataset/arson_detection.yml    server:~/dfine/configs/dataset/
scp configs/dfine/custom/dfine_hgnetv2_n_arson.yml  server:~/dfine/configs/dfine/custom/
```

### Step 7: Server Training

```bash
conda activate arson
cd ~/dfine
python train.py \
  -c configs/dfine/custom/dfine_hgnetv2_n_arson.yml \
  --use-amp
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Config | `dfine_hgnetv2_n_arson.yml` | Nano architecture, 3 classes |
| Pretrained | `dfine_hgnetv2_n_coco.pth` | COCO pretrained weights |
| AMP | Enabled | Mixed precision training |
| Epochs | 80 | Two-stage training |
| Batch size | 32 | Config-defined |
| num_workers | 0 | Required (shared memory limitation) |

### Config Notes

The arson config must explicitly set `num_workers: 0` because D-FINE's `dataloader.yml` defaults to a higher value. Without the override, the server's shared memory limitation causes crashes.

`num_classes: 3` must be set in the dataset config (was 2 in round 1).

### Output Files

```
output/arson_3class_n/
├── best_stg1.pth     # Stage 1 best weights
├── best_stg2.pth     # Stage 2 best weights (= final model)
├── last.pth          # Last epoch weights
└── log.txt           # Training log
```

**Important**: D-FINE uses two-stage training. `best_stg2.pth` is the final model, not `best.pth`.

### Step 8: Deployment

```bash
scp gmission@192.168.1.34:~/dfine/output/arson_3class_n/best_stg2.pth \
    weights/arson_dfine/best.pth
```

The production system loads from `weights/arson_dfine/best.pth`.

## Troubleshooting

See [troubleshooting.md](./troubleshooting.md) for issues encountered during training and inference.