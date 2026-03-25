"""Extract training frames from KISA collapse videos + YOLO auto-labeling
- Before collapse (standing person): GT -30s ~ -5s
- After collapse (lying person): GT +0s ~ +60s
- Extract 1 frame per second (deduplication)
- Auto-label person bbox with YOLO11x
"""
import os, sys, cv2
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO

VIDEO_DIR = Path(r"C:\Users\gmission\Desktop\kisa_vid\falldown")

# (video_file, gt_start_sec) — 10 batch videos
VIDEOS = [
    ("C00_003_0003.mp4", 198),
    ("C00_023_0004.mp4", 197),
    ("C00_034_0005.mp4", 188),
    ("C00_060_0003.mp4", 211),
    ("C00_081_0002.mp4", 236),
    ("C00_146_0004.mp4", 112),
    ("C00_153_0005.mp4", 107),
    ("C00_188_0003.mp4", 128),
    ("C00_217_0004.mp4", 118),
    ("C00_235_0002.mp4", 112),
]

# Extraction interval settings
BEFORE_FALL_START = -30  # from 30s before GT
BEFORE_FALL_END = -5     # until 5s before GT
AFTER_FALL_START = -10   # from 10s before GT (includes falling process)
AFTER_FALL_END = 90      # until 90s after GT
EXTRACT_INTERVAL = 0.5   # extract every 0.5s (2 fps)

YOLO_CONF = 0.15         # low conf to also detect collapsed persons
YOLO_IMGSZ = 960


def extract_frames(video_path, gt_sec, model, output_img_dir, output_lbl_dir, prefix):
    """Extract frames from a single video + YOLO labeling"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Calculate extraction time ranges
    times_before = np.arange(
        max(0, gt_sec + BEFORE_FALL_START),
        max(0, gt_sec + BEFORE_FALL_END),
        EXTRACT_INTERVAL
    )
    times_after = np.arange(
        max(0, gt_sec + AFTER_FALL_START),
        min(duration, gt_sec + AFTER_FALL_END),
        EXTRACT_INTERVAL
    )
    extract_times = list(times_before) + list(times_after)

    count = 0
    for t in extract_times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect person with YOLO
        results = model.predict(
            frame, classes=[0], conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ, verbose=False
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            # Could save frames without person (useful as negative samples)
            # But unlabeled images cause training confusion -> skip
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        H, W = frame.shape[:2]

        # Save image
        phase = "before" if t < gt_sec else "after"
        img_name = f"{prefix}_{phase}_{t:.1f}.jpg"
        img_path = output_img_dir / img_name
        cv2.imwrite(str(img_path), frame)

        # Save YOLO format label (class x_center y_center width height)
        lbl_name = img_name.replace(".jpg", ".txt")
        lbl_path = output_lbl_dir / lbl_name
        with open(lbl_path, "w") as f:
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        count += 1

    cap.release()
    return count


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "datasets" / f"kisa_fall_{ts}"
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  KISA Collapse Frame Extraction + YOLO Auto-Labeling")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    model = YOLO("yolo11x.pt")

    total_count = 0
    for video_file, gt_sec in VIDEOS:
        video_path = VIDEO_DIR / video_file
        if not video_path.exists():
            print(f"  [SKIP] {video_file} — file not found")
            continue

        prefix = video_file.replace(".mp4", "")
        print(f"  [{prefix}] GT={gt_sec}s extracting...", end="", flush=True)

        count = extract_frames(video_path, gt_sec, model, img_dir, lbl_dir, prefix)
        total_count += count
        print(f" {count} frames")

    print(f"\n{'='*60}")
    print(f"  Total {total_count} frames extracted")
    print(f"  Images: {img_dir}")
    print(f"  Labels: {lbl_dir}")
    print(f"{'='*60}\n")

    # Generate data.yaml
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['person']\n")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
