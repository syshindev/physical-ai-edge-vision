# Physical AI Edge Vision

Real-time video surveillance event detection system built during an internship at GMission Inc. for Korea Internet Security Agency (KISA) evaluation.

## Projects
### [Intrusion Detection System](./kisa-intrusion-detection/)
Real-time intrusion detection for KISA video surveillance evaluation. Designed the core detection algorithm including ROI analysis, track-based state management, and event lifecycle logic. Built an end-to-end evaluation framework and improved the system score from **80 to 90+** through systematic parameter tuning.

**Key Work**: Algorithm design, parameter tuning, YOLO finetuning experiment, batch evaluation framework

### [Collapse (Fall) Detection System](./kisa-collapse-detection/)
Real-time fall/collapse detection for KISA video surveillance evaluation. Designed a hybrid detection pipeline combining YOLO11x (person detection) with X-CLIP (action classification), featuring a 3-state machine with EMA scoring, multi-evidence verification, and adaptive night mode. Achieved **10/10 PASS** on pre-test.

**Key Work**: Algorithm design (3-state machine, EMA scoring, multi-evidence system), night mode pipeline, tracking recovery system

### [Arson Detection System](./kisa-arson-detection/)
Fire/smoke detection system for KISA arson evaluation. Migrated the detection model from RT-DETR to D-FINE Nano, built the complete training pipeline (dataset conversion, server training, troubleshooting), and integrated the new model into the production system.

**Key Work**: Model migration (RT-DETR → D-FINE), training pipeline, CUDA/config troubleshooting

## Key Achievements
- **30/30 PASS** on KISA intrusion pre-test (30 sample videos, all within -2s ~ +10s tolerance)
- **10/10 PASS** on KISA collapse pre-test with hybrid YOLO + X-CLIP pipeline
- **80 → 90+** score improvement on main evaluation (150 videos) through systematic parameter tuning and algorithm redesign
- Designed adaptive night mode pipeline for collapse detection (near-zero → daytime-level performance)
- Successfully migrated arson detection from RT-DETR to D-FINE Nano with zero regression
- Built complete ML pipeline: frame extraction → annotation (CVAT) → dataset preparation → server training → evaluation
- Resolved 6+ production issues (CUDA errors, config conflicts, score distribution bugs)
- Designed and documented a reusable batch evaluation framework for rapid iteration

## Tech Stack
| Category | Technologies |
|----------|-------------|
| **Detection Models** | YOLO11x, D-FINE Nano (HGNetv2), RT-DETR, X-CLIP (action classification) |
| **Tracking** | BoTSORT (ByteTrack-based multi-object tracker) |
| **Frameworks** | PyTorch, Ultralytics, OpenCV |
| **Data Tools** | CVAT (annotation), custom format converters |
| **Infrastructure** | Linux GPU server, CUDA, conda, tmux |
| **Languages** | Python |

## Related Projects
- [jetson-edge-surveillance](https://github.com/syshindev/jetson-edge-surveillance) — Personal project extending this work to Jetson Orin Nano Super

## About Me
- **Name**: Seungyeop Shin
- **University**: Simon Fraser University
- **Internship**: GMission Inc.
- **Role**: AI Engineer Intern