# Intrusion Detection — Algorithm Design

## ROI Mode Auto-Detection

The system automatically classifies each ROI (Region of Interest) polygon into one of two modes based on its geometric properties. This is critical because thin "fence-line" ROIs and large "area" ROIs require fundamentally different intrusion logic.

### Classification Logic

```
ROI Polygon
    │
    ├── thickness < person_width × 2.5
    │       │
    │       ├── YES → "Strip" mode (thin ROI, e.g., fence line)
    │       │         • Detect crossing events
    │       │         • Lower confirmation delay
    │       │         • Skip full-body-inside check
    │       │
    │       └── NO  → "Area" mode (wide ROI, e.g., parking lot)
    │                 • Detect foot-point entry
    │                 • Higher confirmation delay
    │                 • Full-body-inside ratio check
    │
    └── Additional check: thickness < 100px
            └── YES → Ultra-thin mode
                      • Extended confirmation delay (6s)
                      • More lenient thresholds
```

### Key Metrics

| Metric | Calculation | Purpose |
|--------|-------------|---------|
| `thickness` | `min(w, h)` of minimum bounding rectangle | Determines ROI shape |
| `aspect_ratio` | `max(w, h) / min(w, h)` | Fine-tunes strip behavior |
| `person_width` | Estimated from detected bounding boxes | Adaptive threshold |

## Track State Management

Each detected person is assigned a persistent track ID (via BoTSORT). The system maintains a `TrackState` for each active track.

### Per-Track State Fields

```python
TrackState:
    first_seen: float       # When this person first appeared
    last_seen: float        # Most recent detection time
    last_box: ndarray       # Most recent bounding box

    # Streak counters (for noise filtering)
    foot_in_streak: int     # Consecutive frames with foot inside ROI
    full_in_streak: int     # Consecutive frames with body fully inside
    full_out_streak: int    # Consecutive frames with body outside

    # State flags
    inside_prev: bool       # Was inside ROI in previous frame?
    crossing_time: float    # When crossing was detected (strip mode)
    entered_full_time: float  # When full entry was confirmed
    fully_inside: bool      # Is body fully inside ROI now?

    # Candidate tracking (for recalculation)
    entered_candidate_time: float
    best_candidate_time: float
    best_full_in_streak: int
```

### Foot-Point Detection

The "foot point" represents where a person is standing. It's the bottom-center of the bounding box, optionally offset by a configurable depth value:

```
┌──────────┐
│  Person  │
│   BBox   │
│          │
└────●─────┘  ← foot_point = (center_x, bottom_y + offset)
```

The `FOOT_DEEP_IN_PX` offset is adaptive: `max(10.0, person_width × 0.15)`. This ensures that the foot point is slightly below the visible body boundary, making ROI entry detection more robust to bounding box jitter.

**Strip mode exception**: In strip mode, the reference point shifts from the foot tip to `y2 - h × 0.20` (20% up from bbox bottom). The foot tip detects crossing ~2s too early compared to ground truth, because annotators mark when the body has crossed, not when the foot first touches. Using the lower body as reference naturally scales the delay with walking speed — matching human annotation behavior. See [parameter-tuning.md](./parameter-tuning.md#7-strip-mode-reference-point).

### Inside-Ratio Check

For area-mode ROIs, a person is considered "inside" when their shrunk bounding box corners are at least 85% inside the ROI polygon. The bbox is shrunk by 6% horizontally and 10% vertically to reduce edge noise:

```
Original bbox:          Shrunk bbox:
┌───────────┐           ┌─────────┐
│           │     →     │  ████   │  (6% inset H, 10% inset V)
│           │           │  ████   │
│           │           │  ████   │
└───────────┘           └─────────┘
```

The `inside_ratio_th = 0.85` was tuned from the original 0.98 to prevent false negatives at ROI boundaries.

## Event Lifecycle

### State Machine

```
                    ┌─────────────────────────────────────┐
                    │                                     │
Track enters        ▼                                     │
ROI            ┌──────────┐  CONFIRM_DELAY  ┌──────────┐  │
─────────────► │  START   │ ─ elapsed? ───► │CONFIRMED │  │
               │(pending) │                 │ (active) │  │
               └────┬─────┘                 └────┬─────┘  │
                    │                            │        │
                    │ Track exits ROI            │ Track  │
                    │ before delay               │ exits  │
                    ▼                            ▼        │
               ┌──────────┐            ┌──────────────┐   │
               │CANCELLED │            │     HOLD     │   │
               └──────────┘            │ (grace time) │   │
                                       └──────┬───────┘   │
                                              │           │
                                       Track returns?     │
                                       ├── YES ───────────┘
                                       │
                                       └── NO (timeout)
                                              │
                                              ▼
                                       ┌──────────────┐
                                       │   FINALIZE   │
                                       │ (emit event) │
                                       └──────────────┘
```

### Confirmation Delay

The confirmation delay prevents false positives from brief detections (e.g., a person walking past the ROI boundary):

| ROI Mode | Condition | CONFIRM_DELAY |
|----------|-----------|---------------|
| Area (normal) | — | 1.0s |
| Strip (thin) | `thickness < person_w × 2.5` | 0.0s |
| Ultra-thin | `thickness < 100px` | 6.0s |

### Event Timing Recalculation

When an intrusion event is confirmed, the system may recalculate the start time by looking back at the track history. This handles cases where the person was actually inside the ROI earlier than the initial detection:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MAX_SHIFT` (initial) | 20s (30s if ≤2 confirmed) | Max backward shift for start time |
| `RECALC_MAX_SHIFT` | 8s (12s if roi_aspect > 3.5) | Max shift during recalculation |

### Event Merge

Events with gaps smaller than 10 seconds are merged into a single event. This handles detection flicker — when a person briefly leaves and re-enters the ROI (or tracking momentarily loses them), the system treats it as one continuous intrusion rather than multiple separate events.

```
Event A: 10s──20s    Event B: 25s──40s    (gap = 5s < 10s → merge)
         └──────────────────────┘
Merged:  10s──────────────────40s
```

### Event Selection: Last Event Wins

After merging, the system may still have multiple distinct intrusion events (e.g., two different people entering at different times). The final output selects the **last merged event** (by `raw_start` time), matching KISA's evaluation criteria which judges based on the most recent intrusion.

### NoEvent False-Positive Prevention

For videos where no real intrusion occurs, the finalize logic could still generate false events from brief noise detections. To prevent this, minimum streak thresholds are enforced:

| Parameter | Before | After | Purpose |
|-----------|--------|-------|---------|
| FORCE-CONFIRM streak | 1 | 10 | Prevent single-frame noise from creating events |
| SOFT-CONFIRM streak | 3 | 10 | Require sustained detection before confirming |

These thresholds ensure that only genuine, sustained detections produce events during finalization.

## Multi-Person Handling

When multiple people are tracked simultaneously:

1. Each person maintains independent `TrackState`
2. If one person triggers an event while another is already being tracked, both events are recorded
3. The "last event" selection chooses the most recent one at finalization
4. Multi-person scenarios trigger special recalculation logic for thin ROIs