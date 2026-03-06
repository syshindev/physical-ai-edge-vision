# Intrusion Detection System

Real-time intrusion detection for the KISA video surveillance evaluation program.

## Overview

The system processes live video feeds to detect unauthorized persons entering restricted zones (ROIs). When an intrusion event is detected, it generates a standardized XML report with precise start times. The system must achieve detection within a strict **-2s to +10s** tolerance window relative to the ground truth.

## Architecture

```
  Video Frame
       │
       ▼
┌──────────────┐
│  YOLO 11x    │  Object detection (person class, conf=0.25)
│  imgsz=960   │
└──────┬───────┘
       │  Bounding boxes + confidence
       ▼
┌──────────────┐
│   BoTSORT    │  Multi-object tracking (persistent track IDs)
│   Tracker    │
└──────┬───────┘
       │  Tracked persons with IDs
       ▼
┌──────────────┐
│  ROI Judge   │  Foot-point / inside-ratio / crossing detection
│  (per track) │  Auto-detects area vs strip ROI mode
└──────┬───────┘
       │  Per-track intrusion state
       ▼
┌──────────────┐
│ Event State  │  Start → Hold → End → Finalize
│   Machine    │  Confirmation delay, recalculation, event selection
└──────┬───────┘
       │  Final event
       ▼
┌──────────────┐
│  XML Output  │  KISA-format result with StartTime
└──────────────┘
```

## My Role

- **Legacy Code Redesign**: Analyzed the inherited codebase, identified 6 fundamental architectural limitations, and rebuilt the detection pipeline from scratch. See [legacy-analysis.md](./legacy-analysis.md).
- **Algorithm Design**: Designed the complete intrusion detection pipeline including ROI mode auto-detection, track state management, and event lifecycle logic. See [algorithm-design.md](./algorithm-design.md).
- **Parameter Tuning**: Systematically tuned 6+ parameters to improve score from 80 to 90+ on the national evaluation. See [parameter-tuning.md](./parameter-tuning.md).
- **Data Pipeline**: Built frame extraction, auto-labeling, and dataset preparation tools for YOLO finetuning. See [finetuning-experiment.md](./finetuning-experiment.md).
- **Evaluation Framework**: Created a batch evaluation script that runs all 30 test videos, compares against ground truth, and produces pass/fail reports
- **Troubleshooting**: Diagnosed and fixed issues with thin ROI handling, boundary detection, and event timing. See [troubleshooting.md](./troubleshooting.md).

## Results

| Metric | Value |
|--------|-------|
| Pre-test (30 sample videos) | 30/30 PASS |
| Main evaluation (150 videos) | 80 → 90+ |
| Tolerance | -2s ~ +10s (KISA standard) |
| Model | YOLO11x (pretrained, no finetuning) |
| Inference Size | 960px |

## Key Technical Decisions

1. **Pretrained YOLO11x over finetuned model**: A finetuning experiment with 2,000 labeled frames degraded generalization (30/30 → 27/30). Kept the pretrained model. See [finetuning-experiment.md](./finetuning-experiment.md).

2. **Adaptive ROI modes**: Auto-detects "area" vs "strip" ROI shapes and applies different confirmation delays and thresholds. See [algorithm-design.md](./algorithm-design.md).

3. **Last-event selection**: Changed from "longest event" to "last event" selection to match KISA evaluation criteria. See [parameter-tuning.md](./parameter-tuning.md#6-event-selection-logic).

## Documentation

- [Legacy Analysis](./legacy-analysis.md) — Inherited code problems and redesign decisions
- [Algorithm Design](./algorithm-design.md) — State machine, ROI modes, event lifecycle
- [Parameter Tuning](./parameter-tuning.md) — Score improvement journey with before/after comparisons
- [Finetuning Experiment](./finetuning-experiment.md) — YOLO finetuning attempt and failure analysis
- [Troubleshooting](./troubleshooting.md) — Problems encountered and solutions

## Scripts

| Script | Purpose |
|--------|---------|
| [`batch_eval.py`](./scripts/batch_eval.py) | Run all test videos and produce pass/fail summary |
| [`extract_event_frames.py`](./scripts/extract_event_frames.py) | Extract frames around event timestamps for labeling |
| [`auto_label_person.py`](./scripts/auto_label_person.py) | YOLO11x auto-labeling for CVAT import |
| [`prepare_dataset.py`](./scripts/prepare_dataset.py) | CVAT export → YOLO training dataset |