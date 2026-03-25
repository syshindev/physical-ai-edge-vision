"""Fine-tuning dataset merge: LE2I + KISA + COCO person
- LE2I: fall -> unified as person (class 0)
- KISA: already person (class 0)
- COCO: person images (for generalization)
- train/val split (90/10)
"""
import os, shutil, random
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent

# Source paths
LE2I_DIR = Path(r"C:\Users\gmission\Downloads\Le2i.v1i.yolov11")
KISA_DIR = sorted(PROJECT_ROOT.glob("datasets/kisa_fall_*"))[-1]  # most recent
COCO_DIR = PROJECT_ROOT / "datasets" / "coco_person"
STANDING_DIR = PROJECT_ROOT / "datasets" / "kisa_standing"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "fall_finetune"
TRAIN_IMG = OUTPUT_DIR / "train" / "images"
TRAIN_LBL = OUTPUT_DIR / "train" / "labels"
VAL_IMG = OUTPUT_DIR / "val" / "images"
VAL_LBL = OUTPUT_DIR / "val" / "labels"

VAL_RATIO = 0.1  # 10% validation


def copy_pair(img_src, lbl_src, img_dst_dir, lbl_dst_dir, new_name):
    """Copy an image+label pair"""
    img_ext = img_src.suffix
    shutil.copy2(img_src, img_dst_dir / f"{new_name}{img_ext}")
    if lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst_dir / f"{new_name}.txt")


def collect_le2i():
    """Collect LE2I data (using all of train + valid + test)"""
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = LE2I_DIR / split / "images"
        lbl_dir = LE2I_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def collect_kisa():
    """Collect KISA extracted data"""
    pairs = []
    img_dir = KISA_DIR / "images"
    lbl_dir = KISA_DIR / "labels"
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def collect_coco():
    """Collect COCO person data"""
    pairs = []
    img_dir = COCO_DIR / "images"
    lbl_dir = COCO_DIR / "labels"
    if not img_dir.exists():
        return pairs
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def collect_standing():
    """Standing person data extracted from KISA intrusion videos"""
    pairs = []
    img_dir = STANDING_DIR / "images"
    lbl_dir = STANDING_DIR / "labels"
    if not img_dir.exists():
        return pairs
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def main():
    # Clean up existing output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # Create output directories
    for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect data
    le2i_pairs = collect_le2i()
    kisa_pairs = collect_kisa()
    coco_pairs = collect_coco()
    standing_pairs = collect_standing()

    print(f"LE2I: {len(le2i_pairs)} images")
    print(f"KISA: {len(kisa_pairs)} images")
    print(f"COCO: {len(coco_pairs)} images")
    print(f"Standing: {len(standing_pairs)} images")

    # Merge all
    all_pairs = []
    for i, (img, lbl) in enumerate(le2i_pairs):
        all_pairs.append((img, lbl, f"le2i_{i:05d}"))
    for i, (img, lbl) in enumerate(kisa_pairs):
        all_pairs.append((img, lbl, f"kisa_{i:05d}"))
    for i, (img, lbl) in enumerate(coco_pairs):
        all_pairs.append((img, lbl, f"coco_{i:05d}"))
    for i, (img, lbl) in enumerate(standing_pairs):
        all_pairs.append((img, lbl, f"stand_{i:05d}"))

    print(f"Total: {len(all_pairs)} images")

    # Shuffle + split
    random.shuffle(all_pairs)
    val_n = int(len(all_pairs) * VAL_RATIO)
    val_set = all_pairs[:val_n]
    train_set = all_pairs[val_n:]

    print(f"Train: {len(train_set)} images, Val: {len(val_set)} images")

    # Copy
    for img, lbl, name in train_set:
        copy_pair(img, lbl, TRAIN_IMG, TRAIN_LBL, name)
    for img, lbl, name in val_set:
        copy_pair(img, lbl, VAL_IMG, VAL_LBL, name)

    # Generate data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['person']\n")

    print(f"\ndata.yaml: {yaml_path}")
    print(f"Output: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
