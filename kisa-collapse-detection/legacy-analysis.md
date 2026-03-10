# Collapse Detection — Legacy Code Analysis & Redesign

## Context

I inherited a fall/collapse detection codebase from a previous developer. While it could detect basic falling events in ideal conditions, it had fundamental limitations: no night vision handling, fragile state management that reset on a single missed frame, and no recovery mechanism when the tracker lost a person. I redesigned the entire detection pipeline into a production-grade system with a 3-state machine, EMA-based scoring, multi-evidence verification, and adaptive night mode.

## Legacy System Overview

| Aspect | Legacy (`SimpleFallingDetector`) | Redesigned (`CollapseMonitor`) |
|--------|--------------------------------|-------------------------------|
| **Lines of code** | ~720 | ~1,624 (2.25x) |
| **Classes** | 3 (DeepSortTracker, XCLIPActionClassifier, SimpleFallingDetector) | 2 (XCLIPActionClassifier, CollapseMonitor) |
| **Detection model** | Custom checkpoint + manual NMS | Ultralytics YOLO11x (integrated) |
| **Tracker** | DeepSort (custom `nets.nn`) | BoTSORT (built-in) + ID stitching fallback |
| **Fall logic** | Frame counter + raw XCLIP score | 3-state machine + EMA + multi-evidence |
| **Night handling** | None | 5-stage adaptive (brightness/conf/imgsz/threshold/gate) |
| **Interface** | Standalone CLI (`process_video()`) | Emulator-integrated callback (`process_frame()`) |
| **Parameters** | ~5 hardcoded | ~30 configurable |

## Problems Identified in Legacy Code

### 1. Single-Frame Reset (Fragile State)

```python
# Legacy: one non-fall frame resets everything
if result['is_falling']:
    if person_id not in self.fall_start_time:
        self.fall_start_time[person_id] = current_time
    fall_duration = current_time - self.fall_start_time[person_id]
    if fall_duration >= 5.0:
        self.video_locked = True
else:
    # Instant reset — 5 seconds of progress lost
    del self.fall_start_time[person_id]
```

A person could be clearly fallen on the ground, but if XCLIP returned a slightly different classification for just one frame, the entire 5-second confirmation counter reset to zero.

**Impact**: Missed detections on real falls due to XCLIP score fluctuation.

**Fix**: Replaced with **EMA (Exponential Moving Average) scoring** with alpha=0.75 and a `non_fall_count >= 3` reset threshold. One bad frame no longer destroys accumulated evidence.

### 2. No Physical Evidence Verification

The legacy system relied entirely on XCLIP's classification output. If XCLIP said "falling", it was falling. No cross-validation with physical indicators.

**Impact**: False positives from unusual poses (bending, sitting) and false negatives when XCLIP was uncertain.

**Fix**: Added a **multi-evidence system** that requires both XCLIP score AND physical evidence:
- Y-velocity spike (sudden downward motion)
- Bbox height drop ≥ 30% (person getting shorter)
- Bbox aspect ratio ≥ 0.80 (horizontal orientation)
- XCLIP transition labels (collapsing/falling forward/backward)

### 3. Binary State with No Recovery

```python
# Legacy: two implicit states
self.video_locked = False  # Monitoring
self.video_locked = True   # Done (permanent)
```

Once `video_locked = True`, the system stopped processing entirely. There was no concept of a "suspect" intermediate state or the ability to recover from a false detection.

**Impact**: First false positive locked the entire video. Conversely, if the system missed the fall initially, there was no second chance.

**Fix**: Implemented a **3-state machine** with bidirectional transitions:
```
NORMAL → SUSPECT → CONFIRMED
  ↑         │
  └─────────┘  (recovery if evidence fades)
```

- **NORMAL → SUSPECT**: EMA ≥ threshold + physical evidence + N consecutive frames
- **SUSPECT → CONFIRMED**: 5s sustained + confirmed gate (lying/shape/EMA) + no walk/stand veto
- **SUSPECT → NORMAL**: `non_fall_count ≥ 3` or standing detected for 1.2s

### 4. No Night/Dark Scene Handling

The legacy system used identical parameters regardless of lighting. CCTV footage at night has low contrast, more noise, and smaller apparent person sizes.

**Impact**: Systematic failures on nighttime videos — missed detections due to low confidence scores and noisy bbox estimates.

**Fix**: Added **5-stage adaptive night mode**:

| Stage | Adaptation |
|-------|-----------|
| 1. Detection | Brightness enhancement (CLAHE + scale) + median/Gaussian blur |
| 2. Confidence | `track_conf` 0.10 → 0.06, `fallback_conf` 0.18 → 0.05 |
| 3. Resolution | `imgsz` 960 → 1280 |
| 4. Thresholds | `enter_thr` 0.70 → 0.62, `lie_thr` 0.20 → 0.12 |
| 5. Gates | `shape_gate` AR ≥ 1.35 → 1.10, `ema_gate` ≥ 0.90 → 0.85 |

Night detection uses HSV median + grayscale percentile analysis per frame.

### 5. No Tracking Recovery

When DeepSort lost a person's track ID (occlusion, brief disappearance), the legacy system simply started fresh with a new ID. All accumulated fall evidence was lost.

**Impact**: Falls during occlusion events were missed.

**Fix**: Three-layer tracking recovery:
1. **ID Stitching**: When a new track appears near a recently lost track (IOU ≥ 0.30 or distance ≤ 180px), inherit the old track's state
2. **Fallback Detection**: For SUSPECT/CONFIRMED tracks that lost tracking, crop the last known region and run `model.predict()` to re-detect
3. **Keepalive**: CONFIRMED tracks maintain their last known box for up to 30 seconds even without detection

### 6. XCLIP Underutilization

```python
# Legacy: 3 labels, binary argmax
self.labels = [
    "a person lying on the ground after falling",    # 0
    "a person standing upright",                     # 1
    "a person walking"                               # 2
]
is_falling = (argmax(probs) == 0)  # Binary decision
```

Only 3 labels with argmax-based binary classification. The probability distribution was largely ignored.

**Fix**: Expanded to **6 labels** with probability aggregation:
```python
self.labels = [
    "a person lying on the ground",              # 0 - lie
    "a person collapsing suddenly",              # 1 - collapse
    "a person falling forward to the ground",    # 2 - fall forward
    "a person falling backward to the ground",   # 3 - fall backward
    "a person standing upright",                 # 4 - stand
    "a person walking"                           # 5 - walk
]
fall_score = probs[0] + probs[1] + probs[2] + probs[3]  # Sum of all fall-related
```

This captures the full spectrum of falling poses (lying, collapsing, forward fall, backward fall) and uses the continuous probability sum rather than a binary argmax.

## New Features (Not in Legacy)

| Feature | Purpose |
|---------|---------|
| **PTS Jump Correction** | Stabilize internal timer when video timestamps jump > 2s |
| **Warmup Period** | Suppress noise in first 5 seconds of video |
| **Track GC** | Auto-cleanup by state: NORMAL 2s, SUSPECT 6s, CONFIRMED 30s TTL |
| **Box EMA** | Smooth bbox coordinates during fallback detection (alpha=0.65~0.70) |
| **Confirm Cooldown** | 15s cooldown per track, one confirmation per track |
| **Overlay Latch** | "FALL DETECTED!" text persists 1.5s after state change to prevent flicker |
| **Area Jump Filter** | Reject bbox area increases ≥ 160% as likely false detections |
| **Walk/Stand Veto** | Block CONFIRMED transition if walking or standing is detected |
| **`notify_seek()`** | Reset internal timers when emulator seeks to different position |

## Architecture Comparison

```
Legacy:                            Redesigned:
┌──────────────────┐               ┌──────────────────┐
│ process_video()  │               │ process_frame()  │ ← called per frame
│ (monolithic)     │               └────────┬─────────┘
│                  │                        │
│ read frame       │               ┌────────▼─────────┐
│ detect           │               │ Night Detection  │
│ track            │               │ + Preprocessing  │
│ classify         │               └────────┬─────────┘
│ threshold check  │                        │
│ write XML        │               ┌────────▼─────────┐
│                  │               │ YOLO + BoTSORT   │
└──────────────────┘               │ + ID Stitching   │
                                   └────────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │ Per-Track State  │
                                   │ Machine (×N)     │
                                   │ NORMAL/SUSPECT/  │
                                   │ CONFIRMED        │
                                   └────────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │ XCLIP + EMA      │
                                   │ + Physical       │
                                   │ Evidence         │
                                   └────────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │ Fallback Detect  │
                                   │ + Keepalive      │
                                   └────────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │ Event + XML      │
                                   └──────────────────┘
```

## Key Metrics

| Metric | Legacy | Redesigned |
|--------|--------|------------|
| Detection stability | Resets on 1 missed frame | Tolerates 3+ missed frames |
| Night scene support | None | Full adaptive pipeline |
| Tracking recovery | None | ID stitch + fallback + keepalive |
| False positive control | `video_locked` only | Warmup + veto + cooldown + area filter |
| State granularity | 2 (implicit) | 3 (explicit, bidirectional) |
| Evidence sources | XCLIP only | XCLIP + velocity + bbox shape + height drop |