"""
Auto-label persons using YOLO11x and export in CVAT-importable YOLO format.

Runs YOLO11x inference on all images, keeps only person detections,
and packages the labels into a zip file ready for CVAT import.

Usage:
    python auto_label_person.py \
        --src-dir /path/to/event_frames \
        --output-dir /path/to/output \
        --conf 0.3

Output:
    <output-dir>/
    ├── obj_train_data/    (label .txt files)
    ├── obj.names          (class names)
    ├── obj.data           (dataset metadata)
    ├── train.txt          (image list)
    └── intrusion_labels.zip  <- Import this into CVAT
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

PERSON_CLASS_ID = 0   # COCO person class
IMG_EXTS = (".jpg", ".jpeg", ".png")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11x auto-labeling for CVAT import")
    parser.add_argument("--src-dir", required=True, help="Directory with event frame subdirectories (kisa_*, aihub_*)")
    parser.add_argument("--output-dir", required=True, help="Output directory for labels and zip")
    parser.add_argument("--model", default="yolo11x.pt", help="YOLO model path (default: yolo11x.pt)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    return parser.parse_args()


def collect_images(src_dir):
    """Collect images from kisa_* and aihub_* subdirectories.
    Returns list of (original_path, cvat_filename) tuples."""
    images = []
    for d in sorted(os.listdir(src_dir)):
        full = os.path.join(src_dir, d)
        if not os.path.isdir(full):
            continue
        if not (d.startswith("kisa_") or d.startswith("aihub_")):
            continue
        for f in sorted(os.listdir(full)):
            if f.lower().endswith(IMG_EXTS):
                cvat_name = f"{d}_{f}"
                images.append((os.path.join(full, f), cvat_name))
    return images


def main():
    args = parse_args()
    src_dir = os.path.abspath(args.src_dir)
    out_dir = os.path.abspath(args.output_dir)
    label_dir = os.path.join(out_dir, "obj_train_data")
    os.makedirs(label_dir, exist_ok=True)

    # Collect images
    images = collect_images(src_dir)
    print(f"Total images: {len(images)}")
    if not images:
        print("No images found.")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Run inference and save labels
    total_boxes = 0
    train_list = []

    for img_path, cvat_name in tqdm(images, desc="Auto-labeling"):
        results = model(img_path, conf=args.conf, verbose=False)
        result = results[0]

        stem = Path(cvat_name).stem
        label_file = os.path.join(label_dir, f"{stem}.txt")
        train_list.append(f"obj_train_data/{cvat_name}")

        lines = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != PERSON_CLASS_ID:
                continue
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1

        with open(label_file, "w") as f:
            f.write("\n".join(lines))

    # Write metadata files
    names_file = os.path.join(out_dir, "obj.names")
    with open(names_file, "w") as f:
        f.write("person\n")

    data_file = os.path.join(out_dir, "obj.data")
    with open(data_file, "w") as f:
        f.write("classes = 1\n")
        f.write("train = train.txt\n")
        f.write("names = obj.names\n")
        f.write("backup = backup/\n")

    train_file = os.path.join(out_dir, "train.txt")
    with open(train_file, "w") as f:
        f.write("\n".join(train_list))

    # Create CVAT-importable zip
    zip_path = os.path.join(out_dir, "intrusion_labels.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(names_file, "obj.names")
        zf.write(data_file, "obj.data")
        zf.write(train_file, "train.txt")
        for txt in sorted(os.listdir(label_dir)):
            if txt.endswith(".txt"):
                zf.write(os.path.join(label_dir, txt), f"obj_train_data/{txt}")

    print(f"\nDone!")
    print(f"  Total images: {len(images)}")
    print(f"  Total person bboxes: {total_boxes}")
    print(f"  Labels: {label_dir}")
    print(f"  CVAT import zip: {zip_path}")


if __name__ == "__main__":
    main()