"""Night CCTV frame extraction — for D-FINE night false-positive training
- Extract night frames from KISA videos
- Segments without people: empty labels (negative samples)
- Segments with people: YOLO auto-labels (positive samples)
- Uses both collapse and intrusion videos
"""
import os, sys, cv2, random
import numpy as np
from pathlib import Path
from datetime import datetime

random.seed(42)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO

# Video directories
FALL_DIR = Path(r"C:\Users\gmission\Desktop\kisa_vid\falldown")
INTRUSION_DIR = Path(r"C:\Users\gmission\Desktop\kisa_vid\intrusion")

YOLO_CONF = 0.25
YOLO_IMGSZ = 960
EXTRACT_INTERVAL = 0.3  # extract every 0.3s
DARK_THRESHOLD_V = 105  # based on HSV V channel median


def is_dark_frame(frame):
    """Determine whether a frame is a night frame"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_med = float(np.median(v))
    return v_med < DARK_THRESHOLD_V


def extract_night_frames(video_path, model, img_dir, lbl_dir, prefix, max_per_video=300):
    """Extract night frames from a single video"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    times = np.arange(5.0, duration - 5.0, EXTRACT_INTERVAL)
    random.shuffle(list(times))

    count = 0
    neg_count = 0
    pos_count = 0

    for t in times:
        if count >= max_per_video:
            break

        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Extract night frames only
        if not is_dark_frame(frame):
            continue

        # Detect person with YOLO
        results = model.predict(frame, classes=[0], conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)
        has_person = (results and results[0].boxes is not None and len(results[0].boxes) > 0)

        H, W = frame.shape[:2]
        img_name = f"{prefix}_night_{t:.1f}.jpg"
        cv2.imwrite(str(img_dir / img_name), frame)

        lbl_name = img_name.replace(".jpg", ".txt")
        lbl_path = lbl_dir / lbl_name

        if has_person:
            # positive: person bbox label
            boxes = results[0].boxes.xyxy.cpu().numpy()
            with open(lbl_path, "w") as f:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            pos_count += 1
        else:
            # negative: empty label (no person — teaches D-FINE "no person here")
            with open(lbl_path, "w") as f:
                pass  # empty file
            neg_count += 1

        count += 1

        # augmentation: horizontal flip
        flipped = cv2.flip(frame, 1)
        aug_img_name = f"{prefix}_night_{t:.1f}_flip.jpg"
        cv2.imwrite(str(img_dir / aug_img_name), flipped)
        aug_lbl_name = aug_img_name.replace(".jpg", ".txt")
        aug_lbl_path = lbl_dir / aug_lbl_name
        if has_person:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            with open(aug_lbl_path, "w") as f:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    # horizontal flip: invert x coordinate
                    cx = 1.0 - ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            pos_count += 1
        else:
            with open(aug_lbl_path, "w") as f:
                pass
            neg_count += 1
        count += 1

    cap.release()
    return count, pos_count, neg_count


def main():
    output_dir = PROJECT_ROOT / "datasets" / "night_frames"
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Night CCTV Frame Extraction (for D-FINE training)")
    print(f"{'='*60}\n")

    model = YOLO("yolo11x.pt")

    total = 0
    total_pos = 0
    total_neg = 0

    # Collapse videos
    fall_videos = sorted(FALL_DIR.glob("*.mp4"))
    print(f"  Collapse videos: {len(fall_videos)}")
    for v in fall_videos:
        prefix = v.stem
        print(f"    [{prefix}]", end="", flush=True)
        cnt, pos, neg = extract_night_frames(v, model, img_dir, lbl_dir, prefix)
        total += cnt
        total_pos += pos
        total_neg += neg
        print(f" {cnt} frames (pos:{pos} neg:{neg})")

    # Intrusion videos
    intrusion_videos = sorted(INTRUSION_DIR.glob("*.mp4"))
    print(f"\n  Intrusion videos: {len(intrusion_videos)}")
    for v in intrusion_videos:
        prefix = v.stem
        print(f"    [{prefix}]", end="", flush=True)
        cnt, pos, neg = extract_night_frames(v, model, img_dir, lbl_dir, prefix, max_per_video=150)
        total += cnt
        total_pos += pos
        total_neg += neg
        print(f" {cnt} frames (pos:{pos} neg:{neg})")

    print(f"\n{'='*60}")
    print(f"  Total: {total} frames (positive:{total_pos} negative:{total_neg})")
    print(f"  Images: {img_dir}")
    print(f"  Labels: {lbl_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
