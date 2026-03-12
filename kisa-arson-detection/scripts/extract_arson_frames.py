"""
Extract frames around fire events from KISA arson videos for fire/smoke labeling.

Reads KISA ground-truth XML files to find fire event timestamps, then extracts
frames in a configurable window around each event at a specified sampling rate.

Usage:
    python extract_arson_frames.py \
        --gt-dir /path/to/gt_xmls \
        --vid-dir /path/to/videos \
        --out-dir /path/to/output \
        --window-before 30 --window-after 30 --sample-fps 2
"""

import argparse
import csv
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


def time_str_to_sec(t: str) -> float:
    t = t.strip()
    parts = t.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def parse_gt_xml(xml_path: Path) -> dict | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        alarm = root.find(".//Alarm")
        if alarm is None:
            return None
        st = alarm.find("StartTime")
        dur = alarm.find("AlarmDuration")
        if st is None:
            return None
        start_sec = time_str_to_sec(st.text)
        dur_sec = time_str_to_sec(dur.text) if dur is not None else 20.0
        return {"start_sec": start_sec, "duration_sec": dur_sec}
    except Exception as e:
        print(f"  [WARN] XML parse error {xml_path.name}: {e}")
        return None


def extract_frames(mp4_path: Path, info: dict, out_sub: Path,
                   window_before: int, window_after: int, sample_fps: int) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {mp4_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st_sec = info["start_sec"]
    start_sec = max(0.0, st_sec - window_before)
    end_sec = st_sec + info["duration_sec"] + window_after

    start_frame = int(start_sec * fps)
    end_frame = min(total_frames - 1, int(end_sec * fps))
    step = max(1, int(round(fps / sample_fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    saved = 0
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % step == 0:
            out_path = out_sub / f"frame_{frame_idx:07d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames around fire events from KISA videos")
    parser.add_argument("--gt-dir", required=True, type=Path, help="Directory with ground-truth XML files")
    parser.add_argument("--vid-dir", required=True, type=Path, help="Directory with video files")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for extracted frames")
    parser.add_argument("--window-before", type=int, default=30, help="Seconds before fire event (default: 30)")
    parser.add_argument("--window-after", type=int, default=30, help="Seconds after fire event (default: 30)")
    parser.add_argument("--sample-fps", type=int, default=2, help="Frames per second to extract (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()

    pairs = []
    for xml in sorted(args.gt_dir.glob("*.xml")):
        info = parse_gt_xml(xml)
        if info is None:
            continue
        mp4 = args.vid_dir / (xml.stem + ".mp4")
        if not mp4.exists():
            print(f"  [WARN] Video not found: {mp4.name}")
            continue
        pairs.append((mp4, info))

    if not pairs:
        print("[ERROR] No videos to process.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Arson Frame Extraction")
    print(f"  Videos: {len(pairs)}  |  Window: -{args.window_before}s ~ +{args.window_after}s")
    print(f"  Sampling: {args.sample_fps}fps  |  Output: {args.out_dir}")
    print(f"{'='*60}\n")

    csv_rows = []
    total_saved = 0

    for mp4, info in tqdm(pairs, desc="Extracting"):
        out_sub = args.out_dir / mp4.stem
        if out_sub.exists() and any(out_sub.glob("*.jpg")):
            existing = len(list(out_sub.glob("*.jpg")))
            total_saved += existing
            csv_rows.append({"video": mp4.name, "fire_start": info["start_sec"],
                             "fire_dur": info["duration_sec"], "frames": existing})
            continue

        out_sub.mkdir(exist_ok=True)
        n = extract_frames(mp4, info, out_sub, args.window_before, args.window_after, args.sample_fps)
        total_saved += n
        csv_rows.append({"video": mp4.name, "fire_start": info["start_sec"],
                         "fire_dur": info["duration_sec"], "frames": n})

    csv_path = args.out_dir / "index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "fire_start", "fire_dur", "frames"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDone! {len(pairs)} videos, {total_saved} total frames")
    print(f"Index: {csv_path}")


if __name__ == "__main__":
    main()