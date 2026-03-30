# Collapse Detection — Iteration History

After achieving 10/10 PASS with the base system (YOLO11x + XCLIP), I explored multiple optimization paths to improve detection robustness, visual stability, and model accuracy. This document records what was tried, why it failed, and what was ultimately kept.

## Summary

| Attempt | Goal | Result | Outcome |
|---------|------|--------|---------|
| Visual stabilization (5 patches) | Stable bbox display after CONFIRMED | Side effects (ghost bbox, jitter, new FPs) | **Rolled back** |
| YOLO11x fine-tuning (2 rounds) | Better person detection for fall scenes | 6/10 PASS — domain mismatch | **Abandoned** |
| D-FINE + ByteTrack transition | Higher detection rate, DEXMA-ready | 8/10 PASS — night+snow FPs unsolved | **Rolled back** |
| `_ever_real_tracked` guard | Block FALLBACK-only ghost track FPs | C00_235 FAIL → PASS, no side effects | **Kept** |

**Final state**: 10/10 PASS with original system + `_ever_real_tracked` guard only.

---

## 1. Visual Stabilization Patches — Rolled Back

### Goal

After a person is CONFIRMED as fallen, the bounding box would flicker or disappear when the tracker lost the person briefly. The goal was to keep a stable visual indicator on screen.

### Patches Tried

| Patch | What It Did | Side Effect |
|-------|-------------|-------------|
| Permanent CONFIRMED state | Prevented CONFIRMED → NORMAL transition | Ghost bboxes persisted after person got up |
| RTMPose re-detection | Used pose estimation to re-find CONFIRMED tracks | RTMPose generated skeletons on non-person objects |
| Multi-person FALLBACK | Extended fallback detection to multiple targets | Additional false positives from background objects |
| Bbox EMA smoothing | Smoothed bbox coordinates with exponential average | Bbox lagged behind actual person movement |
| Display state override | Decoupled visual display from internal state | State/display mismatch caused confusing behavior |

### Decision

All 5 patches were rolled back. Each solved the visual flicker but introduced worse problems. Code-level visual stabilization has fundamental limits when the underlying tracker loses the person — the system cannot display what it cannot detect.

---

## 2. YOLO11x Fine-Tuning — Abandoned

### Goal

Improve person detection in fall scenarios (distant persons, post-fall lying poses) by fine-tuning YOLO11x on fall-specific data.

### Dataset (6,361 images)

| Source | Images | Role |
|--------|--------|------|
| LE2I (Roboflow) | 3,010 | Fallen person poses (indoor/lab) |
| COCO person | 2,000 | Standing person generalization |
| KISA standing | 850 | Standing person in actual CCTV |
| KISA fall (CVAT manual review) | 501 | Fallen + standing in actual CCTV |

### Training Config

- Base: `yolo11x.pt` (COCO pretrained)
- Epochs: 15, LR: 0.0005, Freeze: 10, imgsz: 960
- Server: RTX PRO 6000

### Results

| Round | Change | Batch Result | Issue |
|-------|--------|-------------|-------|
| 1st | Full dataset | 6/10 PASS (regression) | LE2I has no standing person labels → false positive increase |
| 2nd | Added standing labels to LE2I | 6/10 PASS (no improvement) | LE2I domain mismatch persists |

### Decision

Abandoned YOLO fine-tuning. LE2I data (indoor lab environments with clean backgrounds) does not generalize to KISA CCTV footage (outdoor, varied lighting, complex backgrounds). Excluded LE2I from all future training data.

**Lesson**: Domain-matched data is more important than data volume. 501 KISA-sourced images are more valuable than 3,010 out-of-domain images.

---

## 3. D-FINE + ByteTrack Transition — Rolled Back

### Goal

Replace YOLO11x + BoTSORT with D-FINE + ByteTrack for higher person detection rates, and align with the DEXMA platform architecture where D-FINE is the shared detector.

### D-FINE vs YOLO Detection Rate (Post-Fall Frames)

| Confidence | YOLO11x | D-FINE (arson model) | D-FINE (augmented) |
|------------|---------|---------------------|-------------------|
| 0.15 | 75% | 98% | 100% |
| 0.30 | 75% | 80% | 90% |
| 0.40 | 75% | — | 90% |

D-FINE showed significantly better person detection, especially at lower confidence thresholds.

### D-FINE Model Training

Extended the existing arson model (person/fire/smoke, 3-class) with additional person data:

| Source | Images |
|--------|--------|
| Arson dataset (existing) | 24,951 |
| COCO person | 2,000 |
| KISA fall frames | 1,370 |
| KISA standing frames | 850 |
| Night frames (negative) | 1,950 |
| **Total** | **31,121** |

Kept 3-class output — goal was a single model for both arson and collapse on DEXMA.

### Implementation

- Replaced YOLO11x + BoTSORT with D-FINE + ByteTrack (supervision library)
- Dynamic confidence: day conf=0.40, night conf=0.35
- Maintained YOLO fallback support

### Batch Result: 8/10 PASS

Two failures on night+snow videos (C00_153, C00_235) — trees and water drops on camera lens detected as persons.

### Night False Positive Mitigation Attempts

| Method | Result |
|--------|--------|
| Temporal consistency filter (skip tracks < 3s) | Insufficient — FP tracks lasted longer |
| RTMPose skeleton validation | RTMPose generates skeletons on non-person objects → ineffective |
| Bbox size filter (min/max) | Caught full-frame bboxes, but not tree/water drop FPs |
| Night detection hysteresis (5s) | Fixed DARK-CHANGE flickering, but not the core FP issue |
| Night negative data training (1,950 images) | C00_153: -96s → -10s improvement, but not fully resolved |

### Decision

Rolled back to original YOLO11x + BoTSORT (10/10 PASS state). Night+snow environments are a model-level limitation — neither D-FINE nor YOLO can reliably distinguish persons from trees/water drops in these conditions.

D-FINE + ByteTrack will be revisited when building the DEXMA CollapseStrategy, where the platform-level context (multi-camera, temporal history) may help address these edge cases.

---

## 4. FALLBACK Ghost Track Fix — Kept

### Problem

C00_235 (night+snow video) produced a -68s false positive. A ghost track with ID -1010 was created purely from FALLBACK detections (conf=0.06~0.31) and reached CONFIRMED state despite never being tracked by YOLO's primary tracker.

### Root Cause

The FALLBACK detection system was designed to re-detect persons that the tracker temporarily lost. But in noisy night scenes, it could create entirely new "tracks" from background noise — tracks that YOLO's primary detection never identified as persons in the first place.

### Fix

Added `_ever_real_tracked` set that records which track IDs have been assigned a positive ID by YOLO's tracker at least once. FALLBACK-only tracks (never seen by primary tracker) are blocked from reaching CONFIRMED state.

### Result

- C00_235: -68s FAIL → -0s PASS
- Full batch: **10/10 PASS**
- No side effects on any other video

This was the only optimization that improved results without introducing regressions. See [troubleshooting.md](./troubleshooting.md#issue-9-fallback-ghost-track-false-positive-c00_235) for the detailed analysis.

---

## Architecture Decision: XCLIP → RTMPose

### Current State (KISA)

```
YOLO11x → person bbox → XCLIP (8-frame action classification) → 3-state machine → event
```

### Target State (DEXMA)

```
D-FINE → person bbox → RTMPose → keypoints → CollapseStrategy
```

### Why RTMPose Over YOLOPose

| | YOLOPose | RTMPose |
|--|---------|---------|
| Approach | Detection + pose in one pass | Top-down: takes bbox, outputs pose only |
| DEXMA fit | Duplicates detection (D-FINE already detects) | Takes D-FINE bbox as input — no duplication |
| Accuracy | Good | Higher (especially body keypoints) |
| Dependency | Ultralytics | mmpose |

RTMPose fits the DEXMA architecture where D-FINE is the shared detector across all strategies (arson, collapse, intrusion). The detector runs once, and each strategy receives person bboxes as input.

### Status

RTMPose transition has been completed and deployed to the DEXMA Watch platform as CollapseStrategy v3.0 (D-FINE + RTMPose-s hybrid). The KISA system remains on YOLO11x + XCLIP (10/10 PASS, no changes planned).

---

## Scripts Created During Iteration

| Script | Purpose |
|--------|---------|
| `batch_eval_collapse.py` | Run all 10 test videos and produce pass/fail summary |
| `extract_fall_frames.py` | Extract frames around fall events from KISA videos + YOLO auto-labeling |
| `extract_standing_frames.py` | Extract standing person frames from KISA intrusion videos |
| `extract_night_frames.py` | Extract night frames (negative samples + left-right flip augmentation) |
| `download_coco_person.py` | Download COCO val2017 person images + generate YOLO labels |
| `merge_fall_dataset.py` | Merge YOLO training data (LE2I + COCO + KISA) |
| `merge_labels_le2i.py` | Augment LE2I labels (add standing person annotations) |
| `prepare_dfine_dataset.py` | Prepare D-FINE training data in COCO format |
| `test_dfine_fall.py` | Compare D-FINE vs YOLO person detection rates on fall frames |
| `train_yolo_fall.py` | YOLO fine-tuning script (`--server` flag for training server) |
| `analyze_collapse_stability.py` | Automated batch log stability analysis |
