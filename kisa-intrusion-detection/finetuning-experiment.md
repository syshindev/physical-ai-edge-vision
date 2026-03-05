# Intrusion Detection — YOLO Finetuning Experiment

## Motivation

The pretrained YOLO11x model occasionally missed persons in challenging conditions (distant views, unusual poses, partial occlusion). The hypothesis was that finetuning on domain-specific data (KISA CCTV footage) would improve detection accuracy.

## Data Pipeline

### 1. Frame Extraction

Extracted frames from event regions of two data sources:

| Source | Videos | Frames | Method |
|--------|--------|--------|--------|
| KISA test set | 30 | 930 (31/video) | GT XML → event time ± 15s, 1fps |
| AI Hub (outsidedoor_01) | 44 | 1,115 | Event XML → time ± 15s, 1fps |
| **Total** | **74** | **2,045** | |

The `extract_event_frames.py` script handles both XML formats (KISA and AI Hub NIA2019) automatically.

### 2. Annotation with CVAT

- Deployed CVAT via Docker (`docker compose` setup)
- Uploaded all 2,045 frames
- Manual person bounding box annotation

**Labeling Guidelines**:
- Pedestrians only (skip persons inside vehicles)
- Partially occluded → bbox covers visible area only
- Silhouettes behind fences → labeled
- Overlapping persons → separate bboxes
- Barely visible (< 10-20% visible) → skipped

**Results**: 1,702 labeled images / 343 empty images / 2,459 total bboxes

### 3. Auto-Labeling (Pre-annotation)

Used `auto_label_person.py` to generate initial labels with YOLO11x, then manually corrected in CVAT. This significantly reduced annotation time.

### 4. Dataset Preparation

`prepare_dataset.py` converts the CVAT export (YOLO 1.1 format) into a training-ready dataset:

```
datasets/intrusion_yolo/
├── images/train/   (1,739 images)
├── images/val/     (306 images)
├── labels/train/
├── labels/val/
└── data.yaml       (single class: person)
```

Split ratio: 85% train / 15% validation (seed=42).

## Training

| Setting | Value |
|---------|-------|
| Server | 192.168.1.34 (GPU workstation) |
| Base Model | yolo11x.pt (pretrained on COCO) |
| Command | `yolo detect train model=yolo11x.pt data=data.yaml epochs=50 imgsz=640 batch=32` |
| Training Time | ~24 minutes |

### Training Metrics (train5)

| Metric | Value |
|--------|-------|
| Precision | 0.986 |
| Recall | 0.937 |
| mAP50 | 0.977 |
| mAP50-95 | 0.812 |

The metrics looked excellent — high precision and recall on the validation set.

## Evaluation Results

### Finetuned Model

| Result | Count | Details |
|--------|-------|---------|
| PASS | 27/30 | 90% pass rate |
| FAIL | 3/30 | See below |

**Failed Videos**:

| Video | Failure Mode | Detail |
|-------|-------------|--------|
| C00_249 | False negative | Person not detected at all |
| C00_255 | Late detection | +61s too late |
| C00_275 | Early detection | -53s too early |

### Pretrained Model (Baseline)

| Result | Count |
|--------|-------|
| PASS | **30/30** |
| FAIL | 0/30 |

## Failure Analysis

The finetuned model performed **worse** than the pretrained model despite excellent training metrics. Root cause analysis:

### 1. Small Dataset Overfitting

With only ~2,000 training images (vs. COCO's 330,000), the model overfitted to the training distribution. KISA test videos have diverse camera angles, lighting conditions, and person appearances that weren't sufficiently represented.

### 2. Domain Shift

The KISA frames and AI Hub frames have different characteristics (resolution, camera height, scene type). The finetuned model may have shifted too strongly toward one distribution at the expense of the other.

### 3. Person Appearance Bias

The limited dataset may have caused the model to become sensitive to specific person appearances (clothing, pose, size) present in the training data, reducing its ability to detect novel persons.

## Decision: Reverted to Pretrained Model

The pretrained YOLO11x model (`yolo11x.pt`) was restored as the production model. The finetuning approach was abandoned for this use case.

## Lessons Learned

1. **Training metrics ≠ production performance**: 0.977 mAP50 on validation doesn't guarantee real-world improvement. Always evaluate on the actual test set.

2. **Small-dataset finetuning is risky**: For detection tasks where the pretrained model already works well, finetuning on < 5,000 images can degrade generalization.

3. **Algorithmic improvements > model changes**: The parameter tuning approach (see [parameter-tuning.md](./parameter-tuning.md)) yielded better results with zero risk of regression.

4. **Build the evaluation pipeline first**: Having the batch evaluation framework made it possible to immediately detect the regression, preventing a bad model from reaching production.