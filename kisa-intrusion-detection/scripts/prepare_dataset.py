"""
Convert CVAT YOLO export + source images into a YOLO training dataset.

Takes a CVAT-exported zip file (YOLO 1.1 format) and the original images,
then splits them into train/val sets with proper directory structure.

Usage:
    python prepare_dataset.py \
        --cvat-zip /path/to/cvat_export.zip \
        --frames-dir /path/to/event_frames \
        --output-dir /path/to/yolo_dataset \
        --val-ratio 0.15

Output:
    <output-dir>/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    ├── labels/val/
    └── data.yaml
"""

import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="CVAT export -> YOLO training dataset")
    parser.add_argument("--cvat-zip", required=True, help="Path to CVAT export zip (YOLO 1.1 format)")
    parser.add_argument("--frames-dir", required=True, help="Directory with event frame subdirectories")
    parser.add_argument("--output-dir", required=True, help="Output YOLO dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def collect_image_map(frames_dir):
    """Build mapping: cvat_stem -> original image path."""
    img_map = {}
    for d in sorted(os.listdir(frames_dir)):
        full = os.path.join(frames_dir, d)
        if not os.path.isdir(full):
            continue
        if not (d.startswith("kisa_") or d.startswith("aihub_")):
            continue
        for f in sorted(os.listdir(full)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                cvat_name = f"{d}_{f}"
                img_map[Path(cvat_name).stem] = os.path.join(full, f)
    return img_map


def main():
    args = parse_args()
    frames_dir = os.path.abspath(args.frames_dir)
    out_dir = os.path.abspath(args.output_dir)

    # Initialize output directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    # Build image mapping
    print("Building image mapping...")
    img_map = collect_image_map(frames_dir)
    print(f"  Mapped images: {len(img_map)}")

    # Extract labels from CVAT zip
    print(f"Reading CVAT zip: {args.cvat_zip}")
    labels = {}
    with zipfile.ZipFile(args.cvat_zip, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".txt") and "obj_train_data" in name:
                stem = Path(name).stem
                content = zf.read(name).decode().strip()
                labels[stem] = content

    print(f"  Label files: {len(labels)}")

    # Match labels to images
    matched = [stem for stem in labels if stem in img_map]
    unmatched = [stem for stem in labels if stem not in img_map]
    print(f"  Matched: {len(matched)}")
    if unmatched:
        print(f"  Unmatched (label without image): {len(unmatched)}")

    if not matched:
        print("No matched images found!")
        sys.exit(1)

    # Train/val split
    random.seed(args.seed)
    random.shuffle(matched)
    val_count = int(len(matched) * args.val_ratio)
    val_set = set(matched[:val_count])
    train_set = set(matched[val_count:])

    print(f"\nSplit: train={len(train_set)}, val={len(val_set)}")

    # Copy files
    stats = {"train": {"imgs": 0, "labeled": 0, "boxes": 0},
             "val":   {"imgs": 0, "labeled": 0, "boxes": 0}}

    for stem in matched:
        split = "val" if stem in val_set else "train"
        src_img = img_map[stem]
        ext = Path(src_img).suffix

        dst_img = os.path.join(out_dir, "images", split, f"{stem}{ext}")
        shutil.copy2(src_img, dst_img)
        stats[split]["imgs"] += 1

        label_content = labels[stem]
        dst_label = os.path.join(out_dir, "labels", split, f"{stem}.txt")
        with open(dst_label, "w") as f:
            f.write(label_content)
        if label_content:
            stats[split]["labeled"] += 1
            stats[split]["boxes"] += len(label_content.split("\n"))

    # Generate data.yaml
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\nnames:\n")
        f.write("  0: person\n")

    print(f"\nDone! Output: {out_dir}")
    print(f"  train: {stats['train']['imgs']} images ({stats['train']['labeled']} labeled, {stats['train']['boxes']} bboxes)")
    print(f"  val:   {stats['val']['imgs']} images ({stats['val']['labeled']} labeled, {stats['val']['boxes']} bboxes)")
    print(f"  data.yaml: {yaml_path}")
    print(f"\nTraining command:")
    print(f"  yolo detect train model=yolo11x.pt data={yaml_path} epochs=50 imgsz=640")


if __name__ == "__main__":
    main()