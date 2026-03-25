"""Prepare dataset for D-FINE training
- Existing arson data (arson_3class, YOLO format) -> COCO JSON
- COCO person data (already downloaded) -> extract person bbox only
- KISA fall + standing data -> COCO JSON
- Merge all -> output in COCO format for D-FINE training

Classes: 0=person, 1=fire, 2=smoke (same as existing arson)
Person-only data has no fire/smoke annotations (empty categories)
"""
import json, shutil, random, cv2
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent

# Source paths
ARSON_DIR = PROJECT_ROOT / "datasets" / "arson_3class"
COCO_PERSON_DIR = PROJECT_ROOT / "datasets" / "coco_person"
KISA_FALL_DIR = sorted(PROJECT_ROOT.glob("datasets/kisa_fall_*"))[-1]
KISA_STANDING_DIR = PROJECT_ROOT / "datasets" / "kisa_standing"
NIGHT_DIR = PROJECT_ROOT / "datasets" / "night_frames"
NIGHT_V2_DIR = PROJECT_ROOT / "datasets" / "night_frames_v2"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "dfine_fall_train"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"

VAL_RATIO = 0.1
CATEGORIES = [
    {"id": 0, "name": "person"},
    {"id": 1, "name": "fire"},
    {"id": 2, "name": "smoke"},
]


def yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h):
    """YOLO normalized → COCO [x, y, width, height] absolute"""
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    bw = w * img_w
    bh = h * img_h
    return [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)]


def load_yolo_pairs(img_dir, lbl_dir):
    """Load YOLO format image+label pairs"""
    pairs = []
    if not img_dir.exists():
        return pairs
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def collect_arson():
    """Collect arson data (all of train + valid)"""
    pairs = []
    for split in ["train", "valid"]:
        img_dir = ARSON_DIR / split / "images"
        lbl_dir = ARSON_DIR / split / "labels"
        pairs.extend(load_yolo_pairs(img_dir, lbl_dir))
    return pairs


def collect_coco_person():
    return load_yolo_pairs(COCO_PERSON_DIR / "images", COCO_PERSON_DIR / "labels")


def collect_kisa_fall():
    return load_yolo_pairs(KISA_FALL_DIR / "images", KISA_FALL_DIR / "labels")


def collect_kisa_standing():
    return load_yolo_pairs(KISA_STANDING_DIR / "images", KISA_STANDING_DIR / "labels")


def collect_night():
    """Collect night data (including empty labels - negative samples)"""
    pairs = []
    for night_dir in [NIGHT_DIR, NIGHT_V2_DIR]:
        img_dir = night_dir / "images"
        lbl_dir = night_dir / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def build_coco_json(pairs, img_dst_dir, prefix_map):
    """Copy images + generate COCO JSON"""
    images = []
    annotations = []
    ann_id = 1

    for idx, (img_path, lbl_path, new_name) in enumerate(pairs):
        # Check image dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Copy image
        dst_name = f"{new_name}.jpg"
        shutil.copy2(img_path, img_dst_dir / dst_name)

        img_id = idx + 1
        images.append({
            "id": img_id,
            "file_name": dst_name,
            "width": w,
            "height": h,
        })

        # Parse labels
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                bbox = yolo_to_coco_bbox(cx, cy, bw, bh, w, h)
                area = bbox[2] * bbox[3]
                if area <= 0:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": bbox,
                    "area": round(area, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

        if (idx + 1) % 2000 == 0:
            print(f"    {idx + 1} images processed...")

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    return coco_json


def main():
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    train_img_dir = TRAIN_DIR / "images"
    val_img_dir = VAL_DIR / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    arson_pairs = collect_arson()
    coco_pairs = collect_coco_person()
    kisa_fall_pairs = collect_kisa_fall()
    kisa_stand_pairs = collect_kisa_standing()
    night_pairs = collect_night()

    print(f"Arson data: {len(arson_pairs)} images")
    print(f"COCO person: {len(coco_pairs)} images")
    print(f"KISA fall: {len(kisa_fall_pairs)} images")
    print(f"KISA standing: {len(kisa_stand_pairs)} images")
    print(f"Night data: {len(night_pairs)} images")

    # Merge all (use prefix to avoid name collisions)
    all_pairs = []
    for i, (img, lbl) in enumerate(arson_pairs):
        all_pairs.append((img, lbl, f"arson_{i:06d}"))
    for i, (img, lbl) in enumerate(coco_pairs):
        all_pairs.append((img, lbl, f"coco_{i:06d}"))
    for i, (img, lbl) in enumerate(kisa_fall_pairs):
        all_pairs.append((img, lbl, f"kfall_{i:05d}"))
    for i, (img, lbl) in enumerate(kisa_stand_pairs):
        all_pairs.append((img, lbl, f"kstand_{i:05d}"))
    for i, (img, lbl) in enumerate(night_pairs):
        all_pairs.append((img, lbl, f"night_{i:05d}"))

    total = len(all_pairs)
    print(f"\nTotal: {total} images")

    # Shuffle + split
    random.shuffle(all_pairs)
    val_n = int(total * VAL_RATIO)
    val_set = all_pairs[:val_n]
    train_set = all_pairs[val_n:]

    print(f"Train: {len(train_set)} images, Val: {len(val_set)} images")

    # Generate COCO JSON
    print("\nGenerating Train COCO JSON...")
    train_json = build_coco_json(train_set, train_img_dir, None)
    with open(TRAIN_DIR / "train.json", "w") as f:
        json.dump(train_json, f)
    print(f"  images: {len(train_json['images'])}, annotations: {len(train_json['annotations'])}")

    print("Generating Val COCO JSON...")
    val_json = build_coco_json(val_set, val_img_dir, None)
    with open(VAL_DIR / "val.json", "w") as f:
        json.dump(val_json, f)
    print(f"  images: {len(val_json['images'])}, annotations: {len(val_json['annotations'])}")

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Train: {TRAIN_DIR}")
    print(f"Val: {VAL_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
