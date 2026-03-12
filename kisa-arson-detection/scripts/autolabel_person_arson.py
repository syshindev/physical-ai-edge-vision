"""
Auto-label person bounding boxes on arson dataset images using YOLO.

Adds person (class 0) labels to existing fire/smoke label files without
overwriting them. Skips images that already have person labels.

Usage:
    python autolabel_person_arson.py \
        --dataset /path/to/arson_3class \
        --weights yolo11s.pt \
        --conf 0.3 --imgsz 640 \
        --prefix kisa_

    # Only process images matching prefix (e.g., KISA frames):
    python autolabel_person_arson.py \
        --dataset /path/to/arson_3class \
        --weights yolo11s.pt \
        --prefix kisa_
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from ultralytics import YOLO


PERSON_CLASS_ID = 0   # YOLO COCO class
PERSON_LABEL_ID = 0   # Target dataset class


def label_path_for(img_path: Path) -> Path:
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def has_person_label(label_file: Path) -> bool:
    if not label_file.exists():
        return False
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == str(PERSON_LABEL_ID):
                return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO person auto-labeling for arson dataset")
    parser.add_argument("--dataset", required=True, type=Path, help="Arson dataset root (with train/valid subdirs)")
    parser.add_argument("--weights", required=True, type=Path, help="YOLO weights file")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--prefix", type=str, default="", help="Only process images with this filename prefix")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.exists():
        print(f"[ERROR] Weights not found: {args.weights}")
        sys.exit(1)

    model = YOLO(str(args.weights))

    images = []
    for split in ("train", "valid"):
        img_dir = args.dataset / split / "images"
        if not img_dir.exists():
            continue
        pattern = f"{args.prefix}*.jpg" if args.prefix else "*.jpg"
        imgs = sorted(img_dir.glob(pattern))
        images.extend(imgs)
        print(f"[{split}] Matching images: {len(imgs)}")

    print(f"Total images: {len(images)}")

    to_process = []
    for img in images:
        if not has_person_label(label_path_for(img)):
            to_process.append(img)
    print(f"Already labeled (skip): {len(images) - len(to_process)}, To process: {len(to_process)}")

    if not to_process:
        print("[INFO] All images already have person labels.")
        return

    total_added = 0
    files_modified = 0

    for i, img_path in enumerate(tqdm(to_process, desc="Person labeling")):
        try:
            results = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, verbose=False)
            boxes = results[0].boxes

            person_lines = []
            for box in boxes:
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue
                cx, cy, w, h = box.xywhn[0].tolist()
                person_lines.append(f"{PERSON_LABEL_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not person_lines:
                continue

            lp = label_path_for(img_path)
            existing = lp.read_text().rstrip("\n") if lp.exists() else ""
            new_content = existing + "\n" + "\n".join(person_lines) + "\n" if existing else "\n".join(person_lines) + "\n"
            lp.write_text(new_content)

            total_added += len(person_lines)
            files_modified += 1
        except Exception as e:
            tqdm.write(f"[ERROR] {img_path.name}: {e}")

        if (i + 1) % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone! Modified {files_modified} files, added {total_added} person bbox.")


if __name__ == "__main__":
    main()