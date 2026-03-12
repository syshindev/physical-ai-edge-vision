"""
Add intrusion detection person data to the arson 3-class dataset.

Copies person-labeled images from the intrusion event frames dataset into
the arson dataset with an 'intr_' prefix. Skips empty labels and already-added images.

Usage:
    python add_intrusion_person_to_arson.py \
        --labels /path/to/intrusion_labels/obj_train_data \
        --frames /path/to/intrusion_event_frames \
        --arson /path/to/arson_3class \
        --train-ratio 0.85

Notes:
    - Class 0 = person (no remapping needed, same in both datasets)
    - Empty label files are skipped
    - 85:15 train/valid split by default
    - Label filename format: kisa_C00_005_0001_frame_0007200.txt
      → maps to: intrusion_event_frames/kisa_C00_005_0001/frame_0007200.jpg
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from tqdm import tqdm


PREFIX = "intr_"


def label_to_image_path(label_file: Path, frames_dir: Path) -> Path | None:
    """Map label filename to source image path.

    Label: kisa_C00_005_0001_frame_0007200.txt
    Image: frames_dir/kisa_C00_005_0001/frame_0007200.jpg
    """
    stem = label_file.stem
    idx = stem.rfind("_frame_")
    if idx == -1:
        return None
    folder = stem[:idx]
    frame_name = stem[idx + 1:]
    return frames_dir / folder / (frame_name + ".jpg")


def parse_args():
    parser = argparse.ArgumentParser(description="Add intrusion person data to arson dataset")
    parser.add_argument("--labels", required=True, type=Path, help="Intrusion labels directory (obj_train_data/)")
    parser.add_argument("--frames", required=True, type=Path, help="Intrusion event frames directory")
    parser.add_argument("--arson", required=True, type=Path, help="Arson 3-class dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Train split ratio (default: 0.85)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.labels.exists():
        print(f"[ERROR] Labels dir not found: {args.labels}")
        sys.exit(1)
    if not args.frames.exists():
        print(f"[ERROR] Frames dir not found: {args.frames}")
        sys.exit(1)

    # Collect valid label-image pairs
    pairs = []
    missing_img = 0
    empty_label = 0

    label_files = sorted(args.labels.glob("*.txt"))
    print(f"Total label files: {len(label_files)}")

    for lf in tqdm(label_files, desc="Scanning labels"):
        if lf.stat().st_size == 0:
            empty_label += 1
            continue
        img_path = label_to_image_path(lf, args.frames)
        if img_path is None or not img_path.exists():
            missing_img += 1
            continue
        pairs.append((img_path, lf))

    print(f"Valid pairs: {len(pairs)}, Empty labels: {empty_label}, Missing images: {missing_img}")
    if not pairs:
        print("[WARN] No valid pairs found.")
        return

    # Skip already-added images
    existing_train = set(f.name for f in (args.arson / "train" / "images").glob(f"{PREFIX}*.jpg"))
    existing_valid = set(f.name for f in (args.arson / "valid" / "images").glob(f"{PREFIX}*.jpg"))
    existing = existing_train | existing_valid
    if existing:
        print(f"[INFO] Already {len(existing)} {PREFIX} images in dataset. Skipping those.")
        pairs = [(img, lbl) for img, lbl in pairs
                 if f"{PREFIX}{lbl.stem}.jpg" not in existing]
        if not pairs:
            print("[INFO] All intrusion images already added.")
            return

    # Split train/valid
    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:split_idx]
    valid_pairs = pairs[split_idx:]

    # Copy files
    for split, split_pairs in [("train", train_pairs), ("valid", valid_pairs)]:
        img_dst = args.arson / split / "images"
        lbl_dst = args.arson / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in tqdm(split_pairs, desc=f"Copying {split}"):
            dst_name = f"{PREFIX}{lbl_path.stem}"
            shutil.copy2(img_path, img_dst / (dst_name + ".jpg"))
            shutil.copy2(lbl_path, lbl_dst / (dst_name + ".txt"))

    total = len(train_pairs) + len(valid_pairs)
    print(f"\nDone! Added {total} intrusion person images.")
    print(f"  Train: +{len(train_pairs)}, Valid: +{len(valid_pairs)}")


if __name__ == "__main__":
    main()