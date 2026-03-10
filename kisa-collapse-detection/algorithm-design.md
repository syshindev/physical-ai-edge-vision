# Collapse Detection — Algorithm Design

## 3-State Machine

The core of the detection system is a per-track state machine with three states:

```
                 EMA >= threshold
                 + physical evidence
                 + N consecutive frames
          ┌──────────────────────────┐
          │                          v
       NORMAL                    SUSPECT ──────────────> CONFIRMED
          ^                       │  │    5s sustained     │
          │   non_fall >= 3       │  │    + confirmed_gate │
          │   or stand 1.2s       │  │    + no walk/stand  │
          └───────────────────────┘  │        veto         │
                                     │                     │
                                     │   stand/walk 1.2s   │
                                     │   or real_lost > 6s │
                                     │         ┌───────────┘
                                     │         v
                                     └──── NORMAL
```

### State Definitions

| State | Meaning | Duration | Exit Condition |
|-------|---------|----------|----------------|
| **NORMAL** | No fall detected | — | EMA + evidence triggers SUSPECT |
| **SUSPECT** | Potential fall, accumulating evidence | 0~5s+ | Confirmed after 5s, or reset if evidence fades |
| **CONFIRMED** | Fall confirmed, event recorded | Hold 5s minimum | Stand/walk detected for 1.2s, or track lost > 6s |

### Why 3 States Instead of 2

The legacy system had only NORMAL and LOCKED (permanent). Problems:
- No way to recover from false SUSPECT detection
- Once locked, entire video processing stopped
- No intermediate "watching closely" phase

The 3-state design allows:
- **Gradual confidence building**: SUSPECT accumulates evidence before committing
- **Recovery from false alarms**: SUSPECT → NORMAL if evidence fades
- **Continued monitoring**: After CONFIRMED, system can detect the person getting back up

## EMA Scoring

### Problem
Raw XCLIP scores fluctuate significantly frame-to-frame. A person clearly lying on the ground might get scores of [0.85, 0.72, 0.91, 0.65, 0.88] across consecutive classifications. The legacy system treated each score independently — one low score reset the entire counter.

### Solution
Exponential Moving Average with alpha=0.75:

```
ema_new = 0.75 * ema_prev + 0.25 * fall_score_current
```

This means ~75% of the score comes from history, providing stability while still responding to changes.

### Decay Rates (When No XCLIP Result Available)

When XCLIP doesn't produce a result (e.g., crop too small, sequence not yet full), the EMA decays at state-dependent rates:

| State | Decay | Reasoning |
|-------|-------|-----------|
| NORMAL | 0.97 | Fast decay — quickly forget stale scores |
| SUSPECT | 0.99 | Slow decay — preserve evidence during brief tracking gaps |
| CONFIRMED | 0.995 | Very slow — maintain confidence through occlusions |

## XCLIP Action Classification

### 6-Label Design

```python
labels = [
    "a person lying on the ground",              # 0 - lie
    "a person collapsing suddenly",              # 1 - collapse (transition)
    "a person falling forward to the ground",    # 2 - fall forward (transition)
    "a person falling backward to the ground",   # 3 - fall backward (transition)
    "a person standing upright",                 # 4 - stand
    "a person walking"                           # 5 - walk
]
```

**Score aggregation**:
```
fall_score = probs[0] + probs[1] + probs[2] + probs[3]   # All fall-related
stand_score = probs[4]
walk_score = probs[5]
```

Labels 1~3 are "transition" labels — they represent the act of falling. Label 0 represents the result (lying). By summing all four, the system captures:
- The initial collapse motion
- Forward/backward falling direction
- The final lying position

### Sequence Sampling

- **Method**: Time-based (every 0.15 seconds), not frame-based
- **Sequence length**: 8 frames per classification
- **Storage**: `deque(maxlen=8)` per track — automatic oldest-frame eviction
- **Crop padding**: `pad_x=15%, pad_y=22%` with state-dependent minimum sizes

## Multi-Evidence System

### SUSPECT Entry Requirements

SUSPECT entry requires **ALL** of:
1. **EMA threshold**: `ema >= 0.70` (night: `>= 0.62`)
2. **Score margins**: `ema >= stand_score + 0.10` AND `ema >= walk_score + 0.25`
3. **Physical evidence** (at least one):
   - `fall_motion_count >= 3` (y-velocity spikes)
   - `h_drop` (bbox height decreased by 30%+)
   - `trans_count >= 3 AND aspect_ratio >= 0.80` (XCLIP transition labels + horizontal bbox)
   - `aspect_ratio >= 1.20` (clearly horizontal/lying)
4. **Consecutive frames**: `enter_count >= 2` (night: `>= 3`)

### CONFIRMED Gate

After 5 seconds in SUSPECT, transition to CONFIRMED requires:
- `ema >= 0.60` (exit threshold)
- **Confirmed gate** (at least one):
  - `lie_gate`: XCLIP "lying" probability >= 0.20 (night: 0.12)
  - `shape_gate`: aspect ratio >= 1.35 (night: 1.10)
  - `ema_gate`: EMA >= 0.90 (night: 0.85)
- **No vetoes**:
  - `walk_veto`: walk_score < 0.45
  - `stand_veto`: stand_score < stand_exit_threshold

### Fast Confirmation (Night)

In dark conditions with very high confidence:
```
if is_dark AND ema >= 0.95 AND lie_prob >= 0.25:
    need_dur = min(need_dur, 2.5)  # Reduce from 5s to 2.5s
```

## Physical Evidence Details

### Y-Velocity EMA
```python
vel_y_ema = 0.7 * vel_y_prev + 0.3 * (dy / dt)
if vel_y_ema > 80.0:    # Night: 60.0
    fall_motion_count += 1
else:
    fall_motion_count = max(0, fall_motion_count - 1)
```

Detects sudden downward movement — the person's center-of-mass dropping rapidly.

### Bbox Height Drop
```python
h_drop = (current_h / previous_h) <= 0.70
```

A 30% height reduction between frames indicates the person went from standing to lying.

### Area Jump Filter (False Positive Prevention)
```python
area_jump = (current_area / previous_area) >= 1.60
if area_jump and aspect_ratio > 1.6:
    enter_evidence = False  # Block SUSPECT entry
```

Sudden area increase with very wide bbox is likely a detection error (two people merging, background object), not a real fall.

## Night Mode Pipeline

### Detection
```python
# HSV + Grayscale analysis
is_dark = (v_median < 105) or (v_p10 < 60) or (gray_mean < 95) or (gray_p10 < 55)
```

### 5-Stage Adaptation

| Stage | Parameter | Day | Night |
|-------|-----------|-----|-------|
| 1. Preprocessing | Brightness | None | CLAHE + scale(1.15) + blur |
| 2. Detection conf | `track_conf` | 0.10 | 0.05~0.06 |
| 3. Resolution | `track_imgsz` | 960 | 1280 |
| 4. Entry threshold | `enter_thr` | 0.70 | 0.62 |
| 5. Confirm gates | `shape_gate` AR | >= 1.35 | >= 1.10 |

### Preprocessing Pipeline (Dark Frames)
```python
det_frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=8)    # Brighten
det_frame = cv2.medianBlur(det_frame, 3)                      # Denoise
det_frame = cv2.GaussianBlur(det_frame, (5, 5), 0)            # Smooth
det_frame = enhance_night(det_frame)                          # CLAHE
```

## Tracking Recovery

### Layer 1: ID Stitching
When a new track appears near a recently lost track:
```python
# IOU-based: score = 0.7 + 0.3 * iou  (if iou >= 0.30)
# Distance-based: score proportional to proximity (if dist <= 180px)
# Accept if score >= 0.30
```
The new track inherits all state (EMA, suspect timer, crops) from the old track.

### Layer 2: Fallback Detection
For SUSPECT/CONFIRMED tracks that lost primary tracking:
```python
# Crop region around last known bbox (padding 35~55%)
roi_frame = frame[ry1:ry2, rx1:rx2]
det = model.predict(roi_frame, conf=fallback_conf)
# Filter by distance and area, apply box EMA to prevent jumps
```

Fallback confidence varies by state:
| Condition | Confidence |
|-----------|-----------|
| Has CONFIRMED track | 0.10 |
| Has SUSPECT track | 0.12 |
| Default | 0.18 |
| Night (with ROI) | 0.05 |
| Night (full frame) | 0.08 |

### Layer 3: Keepalive
When even fallback detection fails, maintain the last known box:
- SUSPECT: keepalive for up to 6 seconds
- CONFIRMED: keepalive for up to 30 seconds
- EMA decays slowly during keepalive (0.999 for CONFIRMED)

## Track Garbage Collection

Stale tracks are cleaned up with state-dependent TTLs:

| State | TTL | Reasoning |
|-------|-----|-----------|
| NORMAL | 2s | Quick cleanup, low cost to re-detect |
| SUSPECT | 6s | Preserve evidence through brief occlusions |
| CONFIRMED | 30s | Maintain confirmed fall as long as possible |

## Event Recording

- **First event wins**: The earliest confirmed fall time is recorded
- **Per-track once**: `confirmed_once[track_id]` prevents duplicate confirmations
- **Cooldown**: 15-second cooldown between confirmations
- **Time source**: Raw elapsed time (PTS-corrected) for accurate XML timestamps

## PTS Jump Correction

Video timestamps can jump unexpectedly (seek, stream restart):
```python
if dt_raw > 2.0:  # Jump detected
    dt_use = 1.0 / max(1.0, fps)  # Use single frame duration instead
corrected_time += dt_use
```

This prevents the internal timer from jumping, which would corrupt duration-based state transitions.