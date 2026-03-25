"""Extract standing/walking person frames from KISA intrusion videos + YOLO auto-labeling
- Extract frames with people from 30 intrusion videos
- Auto-label person bbox with YOLO11x
- Standing person data -> for maintaining generalization during fine-tuning
"""
import sys, cv2, random
import numpy as np
from pathlib import Path

random.seed(42)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO

VIDEO_DIR = Path(r"C:\Users\gmission\Desktop\kisa_vid\intrusion")
MAX_TOTAL = 2000      # max total frames to extract
MAX_PER_VIDEO = 100   # max per video
EXTRACT_INTERVAL = 2.0  # extract every 2s
YOLO_CONF = 0.25
YOLO_IMGSZ = 960


def main():
    output_dir = PROJECT_ROOT / "datasets" / "kisa_standing"
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  KISA Intrusion Videos — Standing Person Frame Extraction")
    print(f"{'='*60}\n")

    model = YOLO("yolo11x.pt")

    # Video list
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"  Found {len(videos)} videos\n")

    total_count = 0
    for video_path in videos:
        if total_count >= MAX_TOTAL:
            break

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        prefix = video_path.stem
        vid_count = 0

        # Uniform sampling across the entire video
        times = list(np.arange(5.0, duration - 5.0, EXTRACT_INTERVAL))
        random.shuffle(times)

        print(f"  [{prefix}]", end="", flush=True)

        for t in times:
            if vid_count >= MAX_PER_VIDEO or total_count >= MAX_TOTAL:
                break

            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            results = model.predict(
                frame, classes=[0], conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ, verbose=False
            )

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            H, W = frame.shape[:2]

            # Save image
            img_name = f"{prefix}_{t:.1f}.jpg"
            cv2.imwrite(str(img_dir / img_name), frame)

            # Save YOLO label
            lbl_name = img_name.replace(".jpg", ".txt")
            with open(lbl_dir / lbl_name, "w") as f:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    bw = (x2 - x1) / W
                    bh = (y2 - y1) / H
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            vid_count += 1
            total_count += 1

        cap.release()
        print(f" {vid_count} frames")

    print(f"\n{'='*60}")
    print(f"  Total {total_count} frames extracted")
    print(f"  Images: {img_dir}")
    print(f"  Labels: {lbl_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
