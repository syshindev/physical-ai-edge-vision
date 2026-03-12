"""
Convert YOLO-format annotations to COCO JSON format for D-FINE training.

YOLO format: class_id cx cy w h (normalized 0-1, one .txt per image)
COCO format: JSON with images[], annotations[], categories[]
             bbox: [x, y, w, h] (absolute pixels, top-left origin)

Usage:
    python convert_yolo_to_coco.py \
        --src /path/to/yolo_dataset \
        --dst /path/to/coco_output \
        --classes person fire smoke

    # Default (arson dataset with 3-class: person, fire, smoke):
    python convert_yolo_to_coco.py \
        --src /path/to/arson_3class \
        --dst /path/to/arson_3class_coco

Output:
    <dst>/
    ├── train/
    │   ├── *.jpg (copied images)
    │   └── train.json
    └── val/
        ├── *.jpg
        └── val.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

DEFAULT_CATEGORIES = [
    {"id": 0, "name": "person", "supercategory": "none"},
    {"id": 1, "name": "fire", "supercategory": "none"},
    {"id": 2, "name": "smoke", "supercategory": "none"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO -> COCO JSON converter")
    parser.add_argument("--src", required=True, type=Path, help="Source YOLO dataset root (with train/valid subdirs)")
    parser.add_argument("--dst", required=True, type=Path, help="Output COCO dataset root")
    parser.add_argument("--classes", nargs="+", default=["person", "fire", "smoke"],
                        help="Class names in order (default: person fire smoke)")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Splits to convert (default: auto-detect train/valid/val)")
    return parser.parse_args()


def build_categories(class_names):
    """Build COCO categories list from class names."""
    return [{"id": i, "name": name, "supercategory": "none"}
            for i, name in enumerate(class_names)]


def detect_splits(src_root: Path) -> dict:
    """Auto-detect available splits (train, valid/val)."""
    splits = {}
    for name in ("train", "valid", "val"):
        candidate = src_root / name
        if candidate.exists():
            coco_name = "val" if name == "valid" else name
            splits[coco_name] = candidate
    return splits


def convert_split(split_name: str, src_dir: Path, dst_root: Path,
                  categories: list, valid_class_ids: set):
    """Convert one split to COCO JSON format."""
    img_dir = src_dir / "images"
    lbl_dir = src_dir / "labels"

    dst_img_dir = dst_root / split_name
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    images_list = []
    annotations_list = []
    ann_id = 0

    img_files = sorted(img_dir.glob("*.*"))
    print(f"\n[{split_name}] Converting {len(img_files)} images...")

    for img_id, img_path in enumerate(tqdm(img_files, desc=split_name)):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"  SKIP (cannot open): {img_path.name} - {e}")
            continue

        images_list.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        # Copy image
        dst_img_path = dst_img_dir / img_path.name
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)

        # Parse YOLO label
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                if cls_id not in valid_class_ids:
                    continue  # Filter unexpected classes

                cx_n, cy_n, bw_n, bh_n = map(float, parts[1:5])

                # Normalized center -> absolute top-left
                bw_abs = bw_n * w
                bh_abs = bh_n * h
                x_abs = cx_n * w - bw_abs / 2
                y_abs = cy_n * h - bh_abs / 2

                annotations_list.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x_abs, 2), round(y_abs, 2),
                             round(bw_abs, 2), round(bh_abs, 2)],
                    "area": round(bw_abs * bh_abs, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

    # Save COCO JSON
    coco_dict = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories,
    }

    json_path = dst_root / split_name / f"{split_name}.json"
    with open(json_path, "w") as f:
        json.dump(coco_dict, f)

    print(f"  -> {json_path}")
    print(f"     Images: {len(images_list)}, Annotations: {len(annotations_list)}")

    return len(images_list), len(annotations_list)


def main():
    args = parse_args()

    categories = build_categories(args.classes)
    valid_class_ids = set(range(len(args.classes)))

    print("=" * 60)
    print("YOLO -> COCO JSON Converter")
    print(f"  Source: {args.src}")
    print(f"  Output: {args.dst}")
    print(f"  Classes: {args.classes}")
    print("=" * 60)

    if not args.src.exists():
        print(f"ERROR: Source directory not found: {args.src}")
        sys.exit(1)

    # Detect or use specified splits
    if args.splits:
        splits = {}
        for s in args.splits:
            candidate = args.src / s
            if candidate.exists():
                coco_name = "val" if s == "valid" else s
                splits[coco_name] = candidate
            else:
                print(f"WARNING: Split '{s}' not found at {candidate}")
    else:
        splits = detect_splits(args.src)

    if not splits:
        print("ERROR: No splits found. Expected subdirectories like 'train', 'valid', 'val'.")
        sys.exit(1)

    args.dst.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    total_anns = 0

    for split_name, src_dir in splits.items():
        n_img, n_ann = convert_split(split_name, src_dir, args.dst,
                                     categories, valid_class_ids)
        total_imgs += n_img
        total_anns += n_ann

    print("\n" + "=" * 60)
    print(f"Done! Total images: {total_imgs}, Total annotations: {total_anns}")
    print(f"Output: {args.dst}")
    print("=" * 60)


if __name__ == "__main__":
    main()