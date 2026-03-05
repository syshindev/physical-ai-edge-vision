# Intrusion Detection — Troubleshooting

## Issue 1: Thin ROI False Negatives

**Problem**: Fence-line ROIs (thin strips) were not detecting intrusions because the standard area-based logic required the person's body to be "fully inside" — impossible for a narrow strip.

**Root Cause**: The same ROI judgment logic was applied to all ROI shapes. A thin strip ROI can never contain a full person bounding box.

**Solution**: Implemented automatic ROI mode detection based on polygon thickness relative to person width. Strip-mode ROIs use crossing detection (foot-point transitions from outside to inside) instead of full-body containment.

```
Area ROI:                    Strip ROI:
┌──────────────┐             ┌──┐
│              │             │  │
│   Person     │             │P │  ← Person crosses the line
│   inside     │             │  │
│              │             └──┘
└──────────────┘
→ Full-body check            → Crossing check
```

## Issue 2: FP16 Inference Causing Missed Detections

**Problem**: Enabling `half=True` (FP16) for YOLO11x inference to improve speed caused persons to not be detected in some frames.

**Root Cause**: FP16 precision loss in YOLO11x. The model's detection confidence drops below threshold for borderline detections when using half precision.

**Solution**: Removed `half=True`. YOLO11x must run in FP32 mode. The small performance gain wasn't worth the detection degradation.

## Issue 3: Boundary Detection Jitter

**Problem**: Persons standing near the ROI boundary repeatedly triggered enter/exit events, causing incorrect start times.

**Root Cause**: Bounding box coordinates fluctuate frame-to-frame due to detection noise. A person at the ROI edge alternates between "inside" and "outside" states.

**Solution**:
1. Shrink bounding boxes by 6% horizontally and 10% vertically before ROI check (reduces edge sensitivity)
2. Use streak counters (`foot_in_streak`, `full_out_streak`) to require N consecutive frames before state changes
3. Implement hysteresis: once "inside", require a stronger signal to transition to "outside"

## Issue 4: Ultra-Thin ROI Over-Sensitivity

**Problem**: Very thin ROIs (< 100px thick) generated events too quickly because even brief pass-throughs were counted as intrusions.

**Root Cause**: Strip-mode confirmation delay was 0.0s, meaning any single crossing triggered an event immediately.

**Solution**: Added a separate "ultra-thin" mode with a 6.0s confirmation delay. This allows normal foot traffic to pass through thin zones without triggering false positives.

## Issue 5: Event Timing for Distant Persons

**Problem**: Distant persons (appearing very small in frame) were not detected, causing missed events.

**Root Cause**: The minimum bounding box height filter (`min_box_h_ratio = 0.04`) rejected detections where the person was less than 4% of the frame height.

**Solution**: Lowered `min_box_h_ratio` to 0.025. YOLO's confidence threshold provides sufficient filtering — the height filter was an unnecessary additional constraint.

## Issue 6: Multi-Person Event Conflicts

**Problem**: When multiple persons entered the ROI at different times, the system sometimes reported the wrong event (earliest or longest instead of most recent).

**Root Cause**: The event selection logic picked the longest-duration event, but KISA evaluates based on the last intrusion timestamp.

**Solution**: Changed event selection to pick the event with the maximum `raw_start` time (most recent intrusion). This aligned with KISA's evaluation criteria.

## Issue 7: Config File Conflict During Batch Evaluation

**Problem**: Running arson detection tests overwrote `config.xml` with arson video entries, causing intrusion batch evaluation to fail (wrong video list).

**Root Cause**: Both arson and intrusion modules read from the same `config/config.xml` file.

**Solution**: Added a workflow rule: always restore the intrusion video list in `config.xml` after arson testing. Documented this in the team's operational procedures.