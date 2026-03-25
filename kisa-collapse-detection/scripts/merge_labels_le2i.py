"""LE2I label augmentation: merge existing fall labels + YOLO standing person labels
- Run YOLO11x on LE2I images to extract standing person bboxes
- Only add bboxes with low IoU (non-overlapping) against existing fall labels
- Result: fallen person (LE2I) + standing person (YOLO) = complete person labels
"""
import sys, cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO

LE2I_DIR = Path(r"C:\Users\gmission\Downloads\Le2i.v1i.yolov11")
YOLO_CONF = 0.25
YOLO_IMGSZ = 960
IOU_MERGE_TH = 0.3  # If IoU > 0.3 with existing label, consider duplicate -> skip


def iou_xyxy(box1, box2):
    """Calculate IoU of two bboxes (xyxy format)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """YOLO normalized → xyxy absolute"""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """xyxy absolute → YOLO normalized"""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def load_yolo_labels(lbl_path, img_w, img_h):
    """YOLO label file -> xyxy bbox list"""
    boxes = []
    if not Path(lbl_path).exists():
        return boxes
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
    return boxes


def main():
    model = YOLO("yolo11x.pt")

    total_added = 0
    total_images = 0

    for split in ["train", "valid", "test"]:
        img_dir = LE2I_DIR / split / "images"
        lbl_dir = LE2I_DIR / split / "labels"
        if not img_dir.exists():
            continue

        images = sorted(img_dir.glob("*.jpg"))
        print(f"\n[{split}] Processing {len(images)} images...")

        for i, img_path in enumerate(images):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            # Load existing LE2I labels
            existing_boxes = load_yolo_labels(str(lbl_path), img_w, img_h)

            # Detect standing persons with YOLO
            results = model.predict(
                img, classes=[0], conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ, verbose=False
            )

            if not results or results[0].boxes is None:
                total_images += 1
                continue

            yolo_boxes = results[0].boxes.xyxy.cpu().numpy()

            # Add only YOLO bboxes that do not overlap with existing labels
            new_boxes = []
            for ybox in yolo_boxes:
                overlaps = False
                for ebox in existing_boxes:
                    if iou_xyxy(ybox, ebox) >= IOU_MERGE_TH:
                        overlaps = True
                        break
                if not overlaps:
                    new_boxes.append(ybox)

            # Append new bboxes to label file
            if new_boxes:
                with open(lbl_path, "a") as f:
                    for box in new_boxes:
                        cx, cy, w, h = xyxy_to_yolo(
                            box[0], box[1], box[2], box[3], img_w, img_h
                        )
                        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                total_added += len(new_boxes)

            total_images += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(images)}... (+{total_added} bbox)")

    print(f"\n{'='*60}")
    print(f"  Done: {total_images} images processed, {total_added} bboxes added")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
