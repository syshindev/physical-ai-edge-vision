# Collapse (Fall) Detection System

Real-time fall/collapse detection for the KISA video surveillance evaluation program. Detects persons falling and remaining on the ground using a hybrid approach combining object detection (YOLO11x) with video action classification (XCLIP).

## Overview

The system processes live video feeds to detect persons collapsing or falling within designated zones. It must distinguish real falls from similar-looking actions (bending, sitting, crouching) and handle challenging conditions including nighttime footage. When a fall event is confirmed, it generates a standardized XML report.

## Architecture

```
   Video Frame
        │
        ▼
┌────────────────┐
│Night Detection │  HSV + grayscale analysis
│+ Preprocessing │  CLAHE, brightness, blur (if dark)
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   YOLO11x      │  Person detection (conf adaptive: 0.05~0.10)
│   + BoTSORT    │  Tracking with ID stitching fallback
└───────┬────────┘
        │  Tracked persons with bboxes
        ▼
┌────────────────┐
│Fallback Detect │  Re-detect lost SUSPECT/CONFIRMED tracks
│  + Keepalive   │  ROI crop + model.predict() + box EMA
└───────┬────────┘
        │  Per-track bbox stream
        ▼
┌────────────────┐
│ XCLIP Action   │  6-label classification (8-frame sequence)
│  Classifier    │  fall_score = lie + collapse + fall_fwd + fall_bwd
└───────┬────────┘
        │  EMA-smoothed scores
        ▼
┌────────────────┐
│3-State Machine │  NORMAL → SUSPECT → CONFIRMED
│  (per track)   │  Multi-evidence: EMA + velocity + bbox shape
└───────┬────────┘
        │  Confirmed fall event
        ▼
┌────────────────┐
│  XML Output    │  KISA-format result
└────────────────┘
```

## My Role

- **Legacy Code Redesign**: Analyzed the inherited codebase (720 lines, 3 classes), identified 6 fundamental limitations, and rebuilt into a production-grade system (1,624 lines). See [legacy-analysis.md](./legacy-analysis.md).
- **Algorithm Design**: Designed the 3-state machine with EMA scoring, multi-evidence verification, and bidirectional state transitions. See [algorithm-design.md](./algorithm-design.md).
- **Night Mode Pipeline**: Built a 5-stage adaptive system for nighttime footage (brightness enhancement, dynamic confidence/threshold/resolution adjustment). See [algorithm-design.md](./algorithm-design.md#night-mode-pipeline).
- **Tracking Recovery**: Implemented ID stitching, fallback detection, and keepalive mechanisms to maintain detection continuity through occlusions. See [algorithm-design.md](./algorithm-design.md#tracking-recovery).
- **Optimization Iterations**: Explored 3 major optimization paths (visual stabilization, YOLO fine-tuning, D-FINE+ByteTrack transition), analyzed failures, and made data-driven rollback decisions. See [iteration-history.md](./iteration-history.md).
- **FALLBACK Ghost Track Fix**: Identified and fixed a FALLBACK-only false positive pattern on night+snow footage (C00_235). See [troubleshooting.md](./troubleshooting.md#issue-9-fallback-ghost-track-false-positive-c00_235).

## Results

| Metric | Value |
|--------|-------|
| Pre-test (10 sample videos) | 10/10 PASS |
| Main evaluation (150 videos) | Pending |
| Models | YOLO11x (person detection) + XCLIP (action classification) |
| Inference Size | 960px (night: 1280px) |
| Optimization iterations | 3 major attempts (visual stabilization, YOLO fine-tuning, D-FINE+ByteTrack) — all analyzed and rolled back. See [iteration-history.md](./iteration-history.md) |

## Key Technical Decisions

1. **Hybrid detection approach**: YOLO for person detection + XCLIP for action classification. Neither alone is sufficient — YOLO can't classify actions, and XCLIP alone can't reliably locate persons.

2. **EMA over raw scores**: XCLIP scores fluctuate frame-to-frame. EMA smoothing (alpha=0.75) prevents single-frame misclassification from resetting accumulated evidence.

3. **Multi-evidence requirement**: SUSPECT entry requires BOTH XCLIP score threshold AND physical evidence (velocity spike, bbox height drop, or horizontal aspect ratio). This dramatically reduces false positives from bending/sitting.

4. **Adaptive night mode**: Rather than a single "night threshold", 5 independent parameters adapt to darkness level. This avoids the tradeoff of either missing nighttime falls or over-detecting daytime noise.

5. **6-label XCLIP over 3-label**: The original 3 labels (lying/standing/walking) struggled with forward falls — the motion pattern was too different from sideways/backward falls to be captured by a single "lying" label. Expanded to 6 labels (lying/collapsing/falling forward/falling backward/standing/walking) and summed all fall-related probabilities. This reliably detected forward falls without regressing on other directions.

## Documentation

- [Legacy Analysis](./legacy-analysis.md) — Inherited code problems and redesign decisions
- [Algorithm Design](./algorithm-design.md) — 3-state machine, EMA scoring, night mode, evidence system
- [Iteration History](./iteration-history.md) — Optimization attempts, failures, and rollback decisions
- [Troubleshooting](./troubleshooting.md) — 9 issues documented (XCLIP flickering, night scenes, ghost tracks, etc.)

## Scripts

| Script | Purpose |
|--------|---------|
| [`batch_eval_collapse.py`](./scripts/batch_eval_collapse.py) | Run all 10 test videos and produce pass/fail summary |
| [`extract_fall_frames.py`](./scripts/extract_fall_frames.py) | Extract frames around fall events + YOLO auto-labeling |
| [`extract_standing_frames.py`](./scripts/extract_standing_frames.py) | Extract standing person frames from intrusion videos |
| [`extract_night_frames.py`](./scripts/extract_night_frames.py) | Extract night frames (negative samples + flip augmentation) |
| [`download_coco_person.py`](./scripts/download_coco_person.py) | Download COCO val2017 person images + YOLO labels |
| [`merge_fall_dataset.py`](./scripts/merge_fall_dataset.py) | Merge YOLO training data (LE2I + COCO + KISA) |
| [`merge_labels_le2i.py`](./scripts/merge_labels_le2i.py) | Augment LE2I labels (add standing person annotations) |
| [`prepare_dfine_dataset.py`](./scripts/prepare_dfine_dataset.py) | Prepare D-FINE training data in COCO format |
| [`test_dfine_fall.py`](./scripts/test_dfine_fall.py) | Compare D-FINE vs YOLO detection rates on fall frames |
| [`train_yolo_fall.py`](./scripts/train_yolo_fall.py) | YOLO fine-tuning script (`--server` flag for training server) |
| [`analyze_collapse_stability.py`](./scripts/analyze_collapse_stability.py) | Automated batch log stability analysis |
