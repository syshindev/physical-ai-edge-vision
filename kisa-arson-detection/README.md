# Arson Detection System

Fire and smoke detection for the KISA (Korea Internet & Security Agency) national CCTV surveillance evaluation program.

## Overview

The system detects fire/smoke events in CCTV footage and reports them following KISA's specific arson rules. It uses a custom-trained D-FINE Nano model (3-class: person, fire, smoke) with a fire state machine, dynamic day/night thresholds, and gap-based event selection.

## Architecture

```
    Video Frame
         │
         ▼
┌────────────────────┐
│  Night Detection   │  First 30 frames: brightness < 120 or person noise > 50%
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│    D-FINE Nano     │  Single forward pass → person + fire + smoke
│   (HGNetv2 B0)     │  Dynamic thresholds (day/night)
│    imgsz=640       │
└─────────┬──────────┘
          │
          ├── Person boxes ──► Rule #1: Person-in-zone check
          │                    (3-frame consecutive confirmation)
          │                    (conf ≥ 0.3 filter, >20 per frame = noise skip)
          │
          ├── Fire boxes ──┐
          │                │
          └── Smoke boxes ─┤
                           ▼
                  Fire State Machine
                  ├── Confirm: day=5 / night=30 consecutive frames
                  ├── Release: 5 consecutive non-detections + count=0
                  ├── Miss: day=soft decay(-1) / night=hard reset(=0)
                  └── Merge: gap < 1.0s → combine events
                           │
                           ▼
                  ┌─────────────────┐
                  │ Event Selection │
                  │   (gap-based)   │
                  │ gap ≤ 15s: same │
                  │ gap > 15s: skip │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   KISA Rules    │
                  │ #1: Person gate │
                  │ #2: dur ≥ 2s    │
                  │ #3: dur < 2s    │
                  │ StartTime =     │
                  │  fire_time+10s  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   XML Output    │
                  └─────────────────┘
```

### Dynamic Day/Night Threshold System

The system automatically detects nighttime conditions and adjusts all thresholds to handle noise:

| Parameter | Day | Night | Reason |
|-----------|-----|-------|--------|
| `fire_conf` | 0.30 | 0.50 | Night noise produces low-conf false positives |
| `fire_confirm_frames` | 5 | 30 | 1-second burst noise can pass 15 frames |
| `person_conf` | 0.05 | 0.15 | Night noise triggers many false person detections |
| `person_draw_threshold` | ≥ 0.3 | ≥ 0.5 | Reduce visual noise in output |
| `confirm_count` on miss | Soft decay (-1) | Hard reset (=0) | Night noise bursts need complete reset |

**Night detection**: Two-stage check in first 30 frames:
1. **Brightness**: Frame mean brightness < 120 → immediate night mode
2. **Noise ratio**: If >50% of frames have >20 person detections → night mode (handles cases where brightness is borderline but noise is high)

### KISA Arson Rules

| Rule | Condition | Action |
|------|-----------|--------|
| **#1** (Required) | Person detected in zone at least once | Gate: must pass to generate any event |
| **#2** | Fire duration ≥ 2 seconds | StartTime = first_fire_time + 10s |
| **#3** | Fire duration < 2s but Rule #1 passes | StartTime = first_fire_time + 10s |
| **Special** | Only the first fire event is recorded | Subsequent fires are logged but not reported |

## My Role

- **Model Migration**: Replaced RT-DETR with D-FINE Nano — new model wrapper, config changes, tracker removal
- **3-Class Redesign**: Separated fire_smoke into independent fire and smoke classes with per-class confidence thresholds
- **Night Mode System**: Designed automatic day/night detection with dynamic threshold adjustment (soft decay vs hard reset)
- **Data Pipeline**: Built end-to-end pipeline — frame extraction, YOLO auto-labeling, CVAT manual review, Roboflow augmentation, COCO conversion (24,000+ total images across 3 training rounds)
- **Iterative Training**: 3 rounds of D-FINE training with incremental data augmentation (1st: domain mismatch failure, 2nd: 3-class success, 3rd: person data augmentation)
- **Event Selection Logic**: Designed gap-based event selection to distinguish false-positive flicker from real fire events
- **Troubleshooting**: Resolved production issues (CUDA errors, class ID conflicts, nighttime noise, camera water drops, D-FINE imgsz constraints)

## Results

| Metric | Value |
|--------|-------|
| Model | D-FINE Nano (HGNetv2 B0), 58MB |
| Training Data | 21,901 train / 2,516 val images (640x640) |
| Classes | person (0), fire (1), smoke (2) |
| Data Sources | Roboflow CCTV fire/smoke, KISA arson frames, intrusion person data |
| Batch Test | **10/10 PASS** (tolerance range: +0s ~ +9s) |
| Detection Tolerance | -2s ~ +10s (KISA standard) |

### Training History

| Round | Data | Result | Issue |
|-------|------|--------|-------|
| 1st | 8,695 images, 2-class (person/fire_smoke) | FAIL | Domain mismatch — internet fire images vs KISA CCTV |
| 2nd | 21,901 images, 3-class + Roboflow CCTV data | PASS | 9/10, nighttime noise on C00_089 |
| 3rd | +534 person images from intrusion dataset + CVAT review | **10/10 PASS** | Night mode dynamic thresholds solved C00_089 |

## Key Technical Decisions

1. **Dynamic day/night thresholds**: Instead of a single threshold set, the system detects nighttime conditions (low brightness or excessive person detections) and adjusts all confidence thresholds upward. Nighttime uses hard reset (count=0) on miss frames to block noise bursts, while daytime uses soft decay (count-1) to allow flickering small fires to accumulate.

2. **Dual-threshold inference**: Person conf=0.05 (distant/small persons in CCTV) vs fire/smoke conf=0.30 (daytime). Model runs at min(conf) and filters per-class in post-processing.

3. **Gap-based event selection**: Multiple fire events are analyzed by gap distance. Events within 15s of each other are treated as the same fire; isolated early events are treated as false positives.

4. **Person consecutive confirmation**: Requires 3 consecutive frames of person detection with conf ≥ 0.3 to confirm (prevents single-frame nighttime noise). Frames with >20 person detections are skipped entirely as noise.

5. **Cross-domain person data**: Reused 534 manually-labeled intrusion detection images to augment person training data for arson, avoiding redundant labeling effort.

6. **D-FINE imgsz=640 fixed**: Transformer positional embeddings are tied to training resolution. Changing imgsz at inference causes dimension mismatch errors.

## Documentation

- [Model Migration](./model-migration.md) — RT-DETR → D-FINE Nano migration process
- [Training Pipeline](./training-pipeline.md) — 3-round training with incremental data augmentation
- [Troubleshooting](./troubleshooting.md) — 9 issues documented (CUDA, nighttime, domain mismatch, etc.)

## Scripts

| Script | Purpose |
|--------|---------|
| [`convert_yolo_to_coco.py`](./scripts/convert_yolo_to_coco.py) | Convert YOLO-format labels to COCO JSON for D-FINE training |
| [`extract_arson_frames.py`](./scripts/extract_arson_frames.py) | Extract frames around fire events from KISA videos |
| [`autolabel_person_arson.py`](./scripts/autolabel_person_arson.py) | YOLO auto-labeling for person class on arson frames |
| [`merge_arson_dataset.py`](./scripts/merge_arson_dataset.py) | Merge multiple data sources into unified 3-class dataset |
| [`add_intrusion_person_to_arson.py`](./scripts/add_intrusion_person_to_arson.py) | Copy intrusion detection person data into arson dataset |
| [`batch_eval_arson.py`](./scripts/batch_eval_arson.py) | Run all 10 test videos and produce pass/fail summary |