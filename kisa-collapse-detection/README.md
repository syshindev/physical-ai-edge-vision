# Collapse (Fall) Detection System

Real-time fall/collapse detection for the KISA video surveillance evaluation program. Detects persons falling and remaining on the ground using a hybrid approach combining object detection (YOLO11x) with video action classification (X-CLIP).

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
│   YOLO 11x     │  Person detection (conf adaptive: 0.05~0.10)
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
│ X-CLIP Action  │  6-label classification (8-frame sequence)
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
- **Night Mode Pipeline**: Built a 5-stage adaptive system for nighttime footage (brightness enhancement, dynamic confidence/threshold/resolution adjustment).
- **Tracking Recovery**: Implemented ID stitching, fallback detection, and keepalive mechanisms to maintain detection continuity through occlusions.

## Results

| Metric | Value |
|--------|-------|
| Pre-test (10 sample videos) | 10/10 PASS |
| Main evaluation (150 videos) | Pending |
| Models | YOLO11x (person detection) + X-CLIP (action classification) |
| Inference Size | 960px (night: 1280px) |

## Key Technical Decisions

1. **Hybrid detection approach**: YOLO for person detection + X-CLIP for action classification. Neither alone is sufficient — YOLO can't classify actions, and X-CLIP alone can't reliably locate persons.

2. **EMA over raw scores**: XCLIP scores fluctuate frame-to-frame. EMA smoothing (alpha=0.75) prevents single-frame misclassification from resetting accumulated evidence.

3. **Multi-evidence requirement**: SUSPECT entry requires BOTH XCLIP score threshold AND physical evidence (velocity spike, bbox height drop, or horizontal aspect ratio). This dramatically reduces false positives from bending/sitting.

4. **Adaptive night mode**: Rather than a single "night threshold", 5 independent parameters adapt to darkness level. This avoids the tradeoff of either missing nighttime falls or over-detecting daytime noise.

5. **6-label XCLIP over 3-label**: The original 3 labels (lying/standing/walking) struggled with forward falls — the motion pattern was too different from sideways/backward falls to be captured by a single "lying" label. Expanded to 6 labels (lying/collapsing/falling forward/falling backward/standing/walking) and summed all fall-related probabilities. This reliably detected forward falls without regressing on other directions.

## Documentation

- [Legacy Analysis](./legacy-analysis.md) — Inherited code problems and redesign decisions
- [Algorithm Design](./algorithm-design.md) — 3-state machine, EMA scoring, night mode, evidence system