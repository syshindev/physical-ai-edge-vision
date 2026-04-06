# Abandonment Detection System

Object abandonment detection for the KISA (Korea Internet & Security Agency) CCTV surveillance evaluation program.

## Overview

The system detects abandoned objects in CCTV footage — a person places an item (bag, trash bag, etc.) and leaves, triggering an event after 10 seconds. It uses a dual detection approach combining MOG2 background subtraction with frame difference (DIFF), featuring automatic day/night mode switching and person path filtering.

## Architecture

```
    Video Frame
         │
         ▼
┌────────────────────┐
│   YOLO11x          │  Person detection (conf=0.25, imgsz=960)
│   + BoTSORT        │  Hysteresis: 30 consecutive frames to confirm
└─────────┬──────────┘
          │  Tracked persons with bboxes
          ▼
┌────────────────────┐
│  Person Path       │  Record heatmap of person movement
│  Heatmap           │  Only check blobs within path radius (150px)
└─────────┬──────────┘
          │
     ┌────┴────┐
     │         │
     ▼         ▼
┌────────┐ ┌────────┐
│  MOG2  │ │  DIFF  │  Dual detection
│ BG sub │ │ frame  │  (background subtraction + frame comparison)
└───┬────┘ └───┬────┘
    │          │
    └────┬─────┘
         ▼
┌────────────────────┐
│  Day/Night Split   │  Brightness ≥ 100: MOG2 primary, weighted average
│                    │  Brightness < 100: DIFF only (MOG2 noise bypass)
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Leave Detection   │  Person > 200px from object OR off-screen
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Event Output      │  event_time = object_placed_time + 10s
│  (XML)             │  KISA tolerance: -2s ~ +10s
└────────────────────┘
```

### Day/Night Mode

| | Day (brightness >= 100) | Night (brightness < 100) |
|---|---|---|
| Detection | MOG2 primary | DIFF only |
| MOG2 | Active (object-type agnostic) | Ignored (noise) |
| DIFF | Supplementary (weighted average) | Primary |
| Noise handling | Person path filter | Full MOG2 bypass |

### Weighted Average (Day Mode)

When both MOG2 and DIFF detect the object:
- Gap > 5s: `0.6 * MOG2_time + 0.4 * DIFF_time`
- Gap <= 5s: MOG2 time adopted directly

MOG2 tends to detect early, DIFF tends to detect late — averaging improves timing accuracy.

## My Role

- **Algorithm Design**: Designed the dual MOG2 + DIFF detection pipeline with automatic day/night mode switching, person path heatmap filtering, and weighted average timing. See [algorithm-design.md](./algorithm-design.md).
- **Iterative Development**: Explored 10 different approaches before reaching 10/10 PASS — from YOLO object tracking to background subtraction to frame difference, systematically narrowing the solution space. See [iteration-history.md](./iteration-history.md).
- **Night Mode Solution**: Identified that MOG2 produces excessive noise in dark/snowy conditions and designed a DIFF-only fallback with hysteresis-based person filtering. See [algorithm-design.md](./algorithm-design.md#daynight-mode-switching).
- **Troubleshooting**: Resolved 13 technical issues including MOG2 tree/shadow noise, uncategorizable object detection, nighttime person flickering, and timing conflicts. See [troubleshooting.md](./troubleshooting.md).
- **Production Deployment**: Ported to DEXMA Watch platform as AbandonmentStrategy (D-FINE + ByteTrack + MOG2/DIFF dual), with 4 parameter adjustments and 3 new logic additions.

## Results

| Metric | Value |
|--------|-------|
| Pre-test (10 sample videos) | **10/10 PASS** (stability verified 2 runs) |
| Main evaluation (150 videos) | Pending |
| Models | YOLO11x (person detection + tracking) |
| Inference Size | 960px |
| Iterations | 10 approaches tried, v10.1 final |
| Detection Tolerance | -2s ~ +10s (KISA standard) |

### Batch Test Results

| Video | Environment | GT | Detected | Diff | Method |
|-------|-------------|-----|----------|------|--------|
| C00_015 | Day | 03:36 | 03:39 | +4s | AVG |
| C00_023 | Day | 03:34 | 03:37 | +4s | AVG |
| C00_028 | Day | 03:31 | 03:32 | +1s | MOG2 |
| C00_036 | Day/distant | 03:29 | 03:37 | +8s | AVG |
| C00_083 | Day | 03:26 | 03:30 | +4s | AVG |
| C00_146 | Night+snow | 01:58 | 02:02 | +4s | DIFF |
| C00_153 | Day | 01:50 | 01:49 | -1s | MOG2 |
| C00_188 | Day | 02:00 | 02:05 | +5s | DIFF |
| C00_221 | Night | 01:51 | 02:00 | +9s | DIFF |
| C00_230 | Day | 01:50 | 01:50 | +0s | AVG |

### Boundary Cases

| Video | Diff | Limit | Margin |
|-------|------|-------|--------|
| C00_221 | +9s | +10s | 1s |
| C00_036 | +8s | +10s | 2s |
| C00_153 | -1s | -2s | 1s |

## Key Technical Decisions

1. **Dual detection (MOG2 + DIFF)**: Neither method alone is sufficient. MOG2 catches objects regardless of type but generates noise from trees/shadows. DIFF catches low-contrast objects MOG2 misses but reacts slowly. Combining both and using weighted average provides the best timing accuracy.

2. **Person path heatmap filtering**: Instead of checking entire frame for abandoned objects, only check areas where a person has walked (150px radius). This dramatically reduces MOG2 false positives from background motion (trees, shadows, animals).

3. **Day/night mode separation**: MOG2 produces uncontrollable noise in dark conditions (snow = foreground after background freeze). Completely bypassing MOG2 at night and relying on DIFF-only detection eliminates this noise class entirely.

4. **Hysteresis person confirmation**: YOLO must detect a person for 30 consecutive frames (~1 second) before confirming as real. This filters out nighttime 1-frame flickering detections and prevents false person path registration.

5. **Distance-based leave detection**: Rather than requiring the person to leave the frame entirely, trigger SUSPECT state when the person moves 200px away from the object. This produces more natural visual transitions while maintaining timing accuracy using frame-exit timestamps.

6. **Weighted average timing**: MOG2 detects 1~5 seconds earlier than DIFF on average. When both detect, using 0.6:0.4 weighted average places the detection time closer to the true placement time, improving accuracy on boundary cases.

## Documentation

- [Algorithm Design](./algorithm-design.md) — Dual detection, day/night mode, hysteresis, weighted average, leave detection
- [Iteration History](./iteration-history.md) — 10 approaches tried, tuning history from v7.0 to v10.1
- [Troubleshooting](./troubleshooting.md) — 13 issues documented (MOG2 noise, night mode, timing conflicts, etc.)

## Scripts

| Script | Purpose |
|--------|---------|
| [`batch_eval_abandonment.py`](./scripts/batch_eval_abandonment.py) | Run all 10 test videos and produce pass/fail summary |
