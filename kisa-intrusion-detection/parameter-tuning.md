# Intrusion Detection — Parameter Tuning

## Overview

The initial system scored in the **low 80s** on the KISA evaluation (150 test videos). Through systematic parameter tuning and algorithm improvements, the score was raised to **90+** while maintaining **30/30 PASS** on the development test set.

## Methodology

1. **Batch evaluation**: Run all 30 test videos through the pipeline
2. **Identify failure modes**: Categorize failures as false negatives (missed), early detection, or late detection
3. **Hypothesize parameter changes**: Based on failure analysis
4. **Test changes**: Re-run batch evaluation, verify no regressions
5. **Iterate**: Repeat until all 30 videos pass

## Parameter Changes

### 1. Inside Ratio Threshold

| | Before | After |
|---|--------|-------|
| **Value** | 0.98 | 0.85 |
| **Problem** | 98% inside requirement was too strict — persons at ROI boundaries were not counted as "inside" even when clearly within the zone |
| **Impact** | Eliminated boundary false negatives |

The `inside_ratio_th` determines what fraction of a person's (shrunk) bounding box corners must be inside the ROI polygon. At 0.98, even a single pixel outside caused rejection. Lowering to 0.85 allows partial boundary overlap while still filtering out people clearly outside.

### 2. Minimum Aspect Ratio

| | Before | After |
|---|--------|-------|
| **Value** | 1.0 | 0.7 |
| **Problem** | Required height > width, which filtered out crouching or bending persons |
| **Impact** | Detected sitting/crouching intrusions |

Standard person detection expects `h > w` (tall bounding box). But crouching people have `h ≈ w` or even `h < w`. Lowering to 0.7 accepts boxes where height is at least 70% of width.

### 3. Confirmation Delay (Area ROI)

| | Before | After |
|---|--------|-------|
| **Value** | 2.0s | 1.0s |
| **Problem** | 2-second delay caused missed detections for fast-moving intrusions |
| **Impact** | Faster response without increasing false positives |

The confirmation delay prevents noise. But 2 seconds was too conservative — some intrusions happened quickly and the person left before confirmation. 1 second still filters jitter while catching fast events.

### 4. Minimum Box Height Ratio

| | Before | After |
|---|--------|-------|
| **Value** | 0.04 | 0.025 |
| **Problem** | Minimum box height = 4% of frame height filtered out distant/small persons |
| **Impact** | Detected intrusions in far-field camera views |

Some KISA test videos have cameras mounted high or aimed at distant areas. Persons appear very small (< 4% of frame height). Lowering to 2.5% captures these without adding noise, since YOLO's confidence threshold already filters non-person detections.

### 5. Foot Deep-In Offset

| | Before | After |
|---|--------|-------|
| **Value** | 10.0px (fixed) | `max(10.0, person_width × 0.15)` |
| **Problem** | Fixed offset didn't scale with person/ROI size |
| **Impact** | More robust entry detection across different camera angles |

The foot-point offset determines how far below the bounding box bottom the "standing position" is assumed to be. A fixed 10px works for medium-distance cameras but fails for close-up or wide-angle views. The adaptive formula scales with the detected person's width.

### 6. Event Merge + Selection Logic

| | Before | After |
|---|--------|-------|
| **Merge** | No merge | Events with gap < 10s merged into one |
| **Selection** | Longest event | Last merged event (`raw_start` maximum) |
| **Problem** | Detection flicker created multiple short events; KISA evaluates the most recent intrusion, not the longest |
| **Impact** | Correct event timing for multi-intrusion videos, robust to brief tracking gaps |

## Results at Each Step

### Step 1: Inference Parameters
```
conf=0.25 + imgsz=960  →  30/30 PASS (no side effects)
```

Previously `imgsz=640`. Upgrading to 960 improved detection of small/distant persons. Tried 1280 but it caused false negatives (excessive detail confused the model).

### Step 2: Algorithm Improvements (5 changes applied together)
```
inside_ratio_th:  0.98 → 0.85
min_aspect:       1.0  → 0.7
CONFIRM_DELAY:    2.0s → 1.0s
min_box_h_ratio:  0.04 → 0.025
FOOT_DEEP_IN_PX:  10.0 → max(10.0, pw×0.15)
→  30/30 PASS (timing also improved)
```

### Step 3: Event Selection Logic
```
Longest event → Last event
→  30/30 PASS (maintained)
```

### Step 4: FP16 Inference Attempt
```
half=True → FALSE NEGATIVES detected → Reverted
```

FP16 (half precision) was tested for speed improvement but caused missed detections. YOLO11x should not use `half=True`.

### 7. Strip Mode Reference Point

| | Before | After |
|---|--------|-------|
| **Value** | `foot_point_xyxy(box, y_offset=6.0)` (bbox bottom) | `y2 - h * 0.20` (20% up from bottom) |
| **Problem** | Strip mode used the foot tip as the crossing reference point. This triggered crossing detection ~2 seconds before the ground truth, because GT annotators mark the moment the person's body has crossed, not when the foot first touches the boundary |
| **Impact** | C00_014_0002 improved from -2s (boundary) to ~0s (safe margin) |

Strip ROIs are thin line/band regions where the person passes through in 1~4 seconds. Using the foot tip as reference detects the crossing at the very first moment of contact. Shifting the reference point to the lower body (20% up from bbox bottom) naturally delays detection proportionally to walking speed — fast crossers get less delay, slow crossers get more — matching human annotation behavior better than a fixed time offset.

This change only affects strip mode (2/30 test videos). Area mode uses a separate code path and is completely unaffected.

### 8. NoEvent False-Positive Prevention

| | Before | After |
|---|--------|-------|
| **FORCE-CONFIRM streak** | 1 | 10 |
| **SOFT-CONFIRM streak** | 3 | 10 |
| **Problem** | Finalize logic generated false events on videos with no real intrusion — brief noise detections were enough to trigger event creation |
| **Impact** | Eliminated false positives on empty (no-event) videos |

## Key Lessons

1. **Start with the simplest change**: Inference size and confidence threshold changes are low-risk and can have significant impact
2. **Batch evaluation is essential**: Individual video testing misses regressions. Always run the full test set
3. **Don't over-optimize**: The finetuning experiment (see [finetuning-experiment.md](./finetuning-experiment.md)) showed that small-dataset finetuning can hurt generalization
4. **Adaptive > fixed thresholds**: Parameters that scale with input characteristics (person size, ROI shape) generalize better than fixed values