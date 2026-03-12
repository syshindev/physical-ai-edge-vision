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
Fire/smoke detection system for KISA arson evaluation. Migrated from RT-DETR to D-FINE Nano with 3-class separation (person/fire/smoke), designed a dynamic day/night threshold system to handle nighttime noise, and built the end-to-end training pipeline across 3 iterative rounds (24,000+ images). Achieved **10/10 PASS** on batch test.

**Key Work**: Model migration (RT-DETR → D-FINE), dynamic day/night thresholds, 3-round iterative training, gap-based event selection

## Key Achievements
- **30/30 PASS** on KISA intrusion pre-test (30 sample videos, all within -2s ~ +10s tolerance)
- **10/10 PASS** on KISA collapse pre-test with hybrid YOLO + X-CLIP pipeline
- **80 → 90+** score improvement on main evaluation (150 videos) through systematic parameter tuning and algorithm redesign
- Designed adaptive night mode pipeline for collapse detection (near-zero → daytime-level performance)
- **10/10 PASS** on arson batch test with dynamic day/night threshold system
- Successfully migrated arson detection from RT-DETR to D-FINE Nano (3-class, 3-round training)
- Built complete ML pipeline: frame extraction → annotation (CVAT) → dataset preparation → server training → evaluation
- Resolved 9+ production issues (CUDA errors, nighttime noise, domain mismatch, config conflicts)
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