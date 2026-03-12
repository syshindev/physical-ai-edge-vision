"""
Merge multiple data sources into a unified 3-class arson dataset.

Combines:
  1. Existing arson dataset (person/fire/smoke, already class-mapped)
  2. KISA CVAT export (fire/smoke labels, remapped from CVAT class IDs)

Output: Unified YOLO-format dataset with train/valid splits.

Usage:
    python merge_arson_dataset.py \
        --existing /path/to/arson_combined \
        --cvat-zip /path/to/cvat_export.zip \
        --cvat-images /path/to/kisa_frames \
        --out /path/to/arson_3class \
        --val-ratio 0.15

Class mapping:
    Unified: 0=person, 1=fire, 2=smoke
    CVAT export: 0=fire→1, 1=smoke→2
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm


CVAT_REMAP = {0: 1, 1: 2}  # CVAT: 0=fire→1, 1=smoke→2


def parse_args():
    parser = argparse.ArgumentParser(description="Merge arson datasets into 3-class unified format")
    parser.add_argument("--existing", required=True, type=Path, help="Existing arson dataset root")
    parser.add_argument("--cvat-zip", required=True, type=Path, help="CVAT YOLO 1.1 export zip")
    parser.add_argument("--cvat-images", required=True, type=Path, help="Directory with KISA frame images")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    out_train_img = args.out / "train" / "images"
    out_train_lbl = args.out / "train" / "labels"
    out_val_img = args.out / "valid" / "images"
    out_val_lbl = args.out / "valid" / "labels"

    for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy existing dataset
    print("1. Copying existing dataset...")
    existing_count = 0
    for split in ["train", "valid"]:
        src_img = args.existing / split / "images"
        src_lbl = args.existing / split / "labels"
        dst_img = out_train_img if split == "train" else out_val_img
        dst_lbl = out_train_lbl if split == "train" else out_val_lbl

        if not src_img.exists():
            continue

        for img in tqdm(sorted(src_img.glob("*.*")), desc=f"  {split}"):
            shutil.copy2(img, dst_img / img.name)
            lbl = src_lbl / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)
            else:
                (dst_lbl / (img.stem + ".txt")).touch()
            existing_count += 1

    print(f"  Existing data: {existing_count} images")

    # 2. Extract and remap CVAT labels
    print("\n2. Remapping CVAT labels...")
    tmp_dir = args.out / "_tmp_cvat"
    tmp_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(args.cvat_zip, "r") as zf:
        zf.extractall(tmp_dir)

    cvat_lbl_dir = tmp_dir / "obj_train_data"
    kisa_img_dir = args.cvat_images.resolve()

    kisa_items = []
    for lbl_file in sorted(cvat_lbl_dir.glob("*.txt")):
        img_path = kisa_img_dir / (lbl_file.stem + ".jpg")
        if not img_path.exists():
            continue

        lines = []
        with open(lbl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id in CVAT_REMAP:
                    parts[0] = str(CVAT_REMAP[cls_id])
                    lines.append(" ".join(parts))

        kisa_items.append((img_path, lines))

    print(f"  KISA images: {len(kisa_items)}")

    # 3. Split KISA data
    random.shuffle(kisa_items)
    val_count = int(len(kisa_items) * args.val_ratio)
    kisa_val = kisa_items[:val_count]
    kisa_train = kisa_items[val_count:]

    for split_name, items, dst_img, dst_lbl in [
        ("train", kisa_train, out_train_img, out_train_lbl),
        ("valid", kisa_val, out_val_img, out_val_lbl),
    ]:
        for img_path, label_lines in tqdm(items, desc=f"  KISA {split_name}"):
            new_name = f"kisa_{img_path.name}"
            shutil.copy2(img_path, dst_img / new_name)
            with open(dst_lbl / f"kisa_{img_path.stem}.txt", "w") as f:
                f.write("\n".join(label_lines))

    shutil.rmtree(tmp_dir)

    # 4. Generate data.yaml
    data_yaml = args.out / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {args.out.resolve()}\n")
        f.write("train: train/images\nval: valid/images\n\n")
        f.write("nc: 3\nnames:\n  0: person\n  1: fire\n  2: smoke\n")

    # 5. Statistics
    train_count = len(list(out_train_img.glob("*.*")))
    val_count = len(list(out_val_img.glob("*.*")))

    print(f"\n{'='*60}")
    print(f"  3-class dataset created!")
    print(f"  Train: {train_count} / Valid: {val_count}")
    print(f"  Output: {args.out.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()