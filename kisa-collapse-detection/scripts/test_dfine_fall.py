"""D-FINE vs YOLO11x fallen person detection comparison test
- Extract post-fall frames from KISA collapse videos
- Detect persons using D-FINE and YOLO11x respectively
- Compare detection success/failure
"""
import os, sys, cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ultralytics import YOLO
from dfine_loader import DFineModel

VIDEO_DIR = Path(r"C:\Users\gmission\Desktop\kisa_vid\falldown")

# Test only post-fall frames (GT + 5~30 seconds)
VIDEOS = [
    ("C00_003_0003.mp4", 198),
    ("C00_023_0004.mp4", 197),
    ("C00_034_0005.mp4", 188),
    ("C00_081_0002.mp4", 236),
    ("C00_146_0004.mp4", 112),
    ("C00_153_0005.mp4", 107),
    ("C00_188_0003.mp4", 128),
    ("C00_235_0002.mp4", 112),
]

TEST_OFFSETS = [5, 10, 15, 20, 30]  # seconds after GT


def main():
    print("Loading models...")
    yolo = YOLO("yolo11x.pt")

    dfine_config = str(PROJECT_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_n_arson.yml")
    dfine_weights = str(PROJECT_ROOT / "weights" / "arson_dfine" / "best.pth")
    dfine = DFineModel(config=dfine_config, weights=dfine_weights, conf=0.15)

    # New D-FINE (person-augmented training) - uses arson config since same model architecture
    dfine_fall_weights = str(PROJECT_ROOT / "weights" / "dfine_fall" / "best.pth")
    dfine_fall = DFineModel(config=dfine_config, weights=dfine_fall_weights, conf=0.40)

    print(f"\n{'Video':<20} {'Time':>6} {'YOLO':>6} {'D-FINE':>6} {'D-FINE+':>6}")
    print("-" * 60)

    yolo_total = 0
    dfine_total = 0
    dfine_fall_total = 0
    total_tests = 0

    for video_file, gt_sec in VIDEOS:
        video_path = VIDEO_DIR / video_file
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        prefix = video_file.replace(".mp4", "")

        for offset in TEST_OFFSETS:
            t = gt_sec + offset
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # YOLO
            yolo_results = yolo.predict(frame, classes=[0], conf=0.15, imgsz=960, verbose=False)
            yolo_count = 0
            if yolo_results and yolo_results[0].boxes is not None:
                yolo_count = len(yolo_results[0].boxes)

            # D-FINE arson (person = class 0)
            dfine_dets = dfine.predict(frame)
            dfine_count = sum(1 for d in dfine_dets if int(d[5]) == 0)

            # D-FINE person-augmented
            dfine_fall_dets = dfine_fall.predict(frame)
            dfine_fall_count = sum(1 for d in dfine_fall_dets if int(d[5]) == 0)

            yolo_total += min(1, yolo_count)
            dfine_total += min(1, dfine_count)
            dfine_fall_total += min(1, dfine_fall_count)
            total_tests += 1

            y_mark = "O" if yolo_count > 0 else "X"
            d_mark = "O" if dfine_count > 0 else "X"
            df_mark = "O" if dfine_fall_count > 0 else "X"
            print(f"{prefix:<20} +{offset:>3}s  {y_mark:>4}({yolo_count})  {d_mark:>4}({dfine_count})  {df_mark:>4}({dfine_fall_count})")

        cap.release()

    print("-" * 60)
    print(f"{'Detection rate':<20} {'':>6} {yolo_total}/{total_tests}   {dfine_total}/{total_tests}   {dfine_fall_total}/{total_tests}")
    print(f"{'':.<20} {'':>6} {yolo_total/total_tests*100:.0f}%    {dfine_total/total_tests*100:.0f}%    {dfine_fall_total/total_tests*100:.0f}%")


if __name__ == "__main__":
    main()
