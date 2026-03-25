"""Extract person images + labels from COCO val2017 (YOLO format)
- Extract only images containing person (class 0) from COCO val2017 annotations
- Generate YOLO format labels
- Sample up to 2000 images
"""
import json, os, random, shutil, urllib.request, ssl
from pathlib import Path
from zipfile import ZipFile

# Bypass SSL verification (COCO site certificate issue)
ssl._create_default_https_context = ssl._create_unverified_context

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "coco_person"
IMG_DIR = OUTPUT_DIR / "images"
LBL_DIR = OUTPUT_DIR / "labels"

COCO_IMG_URL = "https://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"

MAX_IMAGES = 2000
PERSON_CAT_ID = 1  # COCO person category id


def download_if_needed(url, dst):
    if dst.exists():
        print(f"  Already exists: {dst.name}")
        return
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, str(dst))
    print(f"  Done: {dst.name}")


def main():
    cache_dir = PROJECT_ROOT / "datasets" / "_coco_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download
    img_zip = cache_dir / "val2017.zip"
    ann_zip = cache_dir / "annotations_trainval2017.zip"

    print("1. Download COCO val2017")
    download_if_needed(COCO_IMG_URL, img_zip)
    download_if_needed(COCO_ANN_URL, ann_zip)

    # 2. Extract annotations
    print("\n2. Parsing annotations")
    ann_json_path = cache_dir / "annotations" / "instances_val2017.json"
    if not ann_json_path.exists():
        print("  Extracting annotations zip...")
        with ZipFile(ann_zip) as z:
            z.extractall(cache_dir)

    with open(ann_json_path, "r") as f:
        coco = json.load(f)

    # image_id → image info
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Collect person annotations (per image_id)
    person_anns = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != PERSON_CAT_ID:
            continue
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        if img_id not in person_anns:
            person_anns[img_id] = []
        person_anns[img_id].append(ann)

    print(f"  Images containing person: {len(person_anns)}")

    # 3. Sampling
    img_ids = list(person_anns.keys())
    random.shuffle(img_ids)
    selected = img_ids[:MAX_IMAGES]
    print(f"  Sampled: {len(selected)} images")

    # 4. Extract images + labels
    print("\n3. Extracting images + labels")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    img_zipfile = ZipFile(img_zip)
    count = 0

    for img_id in selected:
        img_info = id_to_img[img_id]
        fname = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        # Extract image
        zip_path = f"val2017/{fname}"
        try:
            data = img_zipfile.read(zip_path)
        except KeyError:
            continue

        img_path = IMG_DIR / fname
        with open(img_path, "wb") as f:
            f.write(data)

        # Generate YOLO label
        lbl_path = LBL_DIR / fname.replace(".jpg", ".txt")
        with open(lbl_path, "w") as f:
            for ann in person_anns[img_id]:
                bx, by, bw, bh = ann["bbox"]  # COCO: x,y,w,h (absolute)
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h
                # Range check
                if nw <= 0 or nh <= 0:
                    continue
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        count += 1
        if count % 200 == 0:
            print(f"  {count}/{len(selected)}...")

    img_zipfile.close()

    print(f"\nDone: {count} images")
    print(f"Images: {IMG_DIR}")
    print(f"Labels: {LBL_DIR}")


if __name__ == "__main__":
    main()
