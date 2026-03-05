"""
Extract video frames around event timestamps for labeling.

Supports two XML formats:
  - KISA: <KisaLibraryIndex> with <Alarm><StartTime>
  - AI Hub (NIA2019): <annotation> with <event><starttime>

Extracts frames within +-WINDOW_SEC of each event, at SAMPLE_FPS rate.
Outputs to a flat directory per video with an index CSV for tracking.

Usage:
    python extract_event_frames.py \
        --kisa-gt-dir /path/to/kisa/gt \
        --kisa-vid-dir /path/to/kisa/videos \
        --aihub-dir /path/to/aihub/data \
        --output-dir /path/to/output

    # KISA only:
    python extract_event_frames.py \
        --kisa-gt-dir /path/to/gt --kisa-vid-dir /path/to/vids \
        --output-dir output/frames

    # AI Hub only:
    python extract_event_frames.py \
        --aihub-dir /path/to/aihub \
        --output-dir output/frames
"""

import argparse
import csv
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

WINDOW_SEC = 15       # Extract +-15 seconds around event
SAMPLE_FPS = 1        # Save 1 frame per second
SAVE_W = 1920         # Resize width (only if source is wider)
MARK_KP = True        # Mark AI Hub keypoints with red circles


def parse_args():
    parser = argparse.ArgumentParser(description="Extract event frames for labeling")
    parser.add_argument("--kisa-gt-dir", type=Path, default=None, help="KISA ground truth XML directory")
    parser.add_argument("--kisa-vid-dir", type=Path, default=None, help="KISA video directory")
    parser.add_argument("--aihub-dir", type=Path, default=None, help="AI Hub data directory (mp4+xml pairs)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for extracted frames")
    parser.add_argument("--window", type=int, default=WINDOW_SEC, help=f"Seconds before/after event (default: {WINDOW_SEC})")
    parser.add_argument("--fps", type=int, default=SAMPLE_FPS, help=f"Frames per second to extract (default: {SAMPLE_FPS})")
    return parser.parse_args()


def time_str_to_sec(t: str) -> float:
    """Convert 'HH:MM:SS' or 'HH:MM:SS.f' to float seconds."""
    t = t.strip()
    parts = t.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def parse_kisa_xml(xml_path: Path) -> dict | None:
    """Parse KISA GT XML. Returns event info dict or None."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        alarm = root.find(".//Alarm")
        if alarm is None:
            return None
        st_elem = alarm.find("StartTime")
        if st_elem is None:
            return None

        filename_elem = root.find(".//Filename")
        filename = filename_elem.text.strip() if filename_elem is not None else None

        return dict(
            source="kisa",
            starttime=time_str_to_sec(st_elem.text),
            keyframe=None, kp_x=None, kp_y=None,
            width=1280, height=720, fps=30.0,
            filename=filename,
        )
    except Exception as e:
        print(f"  [WARN] KISA XML parse error {xml_path.name}: {e}")
        return None


def parse_aihub_xml(xml_path: Path) -> dict | None:
    """Parse AI Hub (NIA2019) XML. Returns event info dict or None."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        hdr = root.find("header")
        fps = float(hdr.find("fps").text) if hdr is not None and hdr.find("fps") is not None else 30.0

        event = root.find(".//event")
        if event is None:
            return None
        st_elem = event.find("starttime")
        if st_elem is None:
            return None

        pos = root.find(".//position")
        kf, kp_x, kp_y = None, None, None
        if pos is not None:
            kf_elem = pos.find("keyframe")
            kpx_elem = pos.find("keypoint/x")
            kpy_elem = pos.find("keypoint/y")
            if kf_elem is not None:
                kf = int(kf_elem.text)
            if kpx_elem is not None and kpy_elem is not None:
                kp_x = int(kpx_elem.text)
                kp_y = int(kpy_elem.text)

        return dict(
            source="aihub",
            starttime=time_str_to_sec(st_elem.text),
            keyframe=kf, kp_x=kp_x, kp_y=kp_y,
            width=w, height=h, fps=fps,
            filename=None,
        )
    except Exception as e:
        print(f"  [WARN] AI Hub XML parse error {xml_path.name}: {e}")
        return None


def find_kisa_pairs(gt_dir: Path, vid_dir: Path) -> list[tuple[Path, Path, str]]:
    """Find KISA (video, xml) pairs."""
    pairs = []
    if not gt_dir.exists():
        print(f"[WARN] KISA GT dir not found: {gt_dir}")
        return pairs

    for xml in sorted(gt_dir.rglob("*.xml")):
        mp4_name = xml.stem + ".mp4"
        mp4 = vid_dir / mp4_name
        if not mp4.exists():
            mp4 = xml.with_suffix(".mp4")
        if not mp4.exists():
            found = list(vid_dir.rglob(mp4_name))
            if found:
                mp4 = found[0]

        if mp4.exists():
            pairs.append((mp4, xml, "kisa"))
        else:
            print(f"  [WARN] Video not found: {mp4_name}")
    return pairs


def find_aihub_pairs(aihub_dir: Path) -> list[tuple[Path, Path, str]]:
    """Find AI Hub (mp4, xml) pairs."""
    pairs = []
    if not aihub_dir.exists():
        print(f"[WARN] AI Hub dir not found: {aihub_dir}")
        return pairs

    for mp4 in sorted(aihub_dir.rglob("*.mp4")):
        xml = mp4.with_suffix(".xml")
        if xml.exists():
            pairs.append((mp4, xml, "aihub"))
    return pairs


def extract_frames(mp4_path: Path, info: dict, out_sub: Path,
                   window_sec: int, sample_fps: int) -> int:
    """Extract frames around event time. Returns number of saved frames."""
    fps = info["fps"]
    st_sec = info["starttime"]
    kp_x_src = info.get("kp_x")
    kp_y_src = info.get("kp_y")

    start_sec = max(0.0, st_sec - window_sec)
    end_sec = st_sec + window_sec

    step = max(1, int(round(fps / sample_fps)))

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {mp4_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps > 0:
        fps = actual_fps
        step = max(1, int(round(fps / sample_fps)))

    start_frame = int(start_sec * fps)
    end_frame = min(total_frames - 1, int(end_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % step == 0:
            h_f, w_f = frame.shape[:2]

            if w_f > SAVE_W:
                scale = SAVE_W / w_f
                frame = cv2.resize(frame, (SAVE_W, int(h_f * scale)),
                                   interpolation=cv2.INTER_AREA)
            else:
                scale = 1.0

            # Mark keypoint (AI Hub only)
            if MARK_KP and kp_x_src is not None and kp_y_src is not None:
                kx = int(kp_x_src * scale)
                ky = int(kp_y_src * scale)
                cv2.circle(frame, (kx, ky), 18, (0, 0, 255), 3)
                cv2.circle(frame, (kx, ky), 4, (0, 0, 255), -1)

            out_path = out_sub / f"frame_{frame_idx:07d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1

        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    return saved


def fmt_time(sec: float) -> str:
    """Float seconds to HH:MM:SS.f string."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def main():
    args = parse_args()
    out_dir = args.output_dir
    window_sec = args.window
    sample_fps = args.fps

    all_pairs = []

    if args.kisa_gt_dir and args.kisa_vid_dir:
        kisa_pairs = find_kisa_pairs(args.kisa_gt_dir, args.kisa_vid_dir)
        print(f"[KISA]   {len(kisa_pairs)} videos found")
        all_pairs.extend(kisa_pairs)

    if args.aihub_dir:
        aihub_pairs = find_aihub_pairs(args.aihub_dir)
        print(f"[AI Hub] {len(aihub_pairs)} videos found")
        all_pairs.extend(aihub_pairs)

    total = len(all_pairs)
    if total == 0:
        print("[ERROR] No videos to process.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Event Frame Extraction")
    print(f"  Videos: {total}  |  Window: +-{window_sec}s  |  {sample_fps}fps")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}\n")

    csv_rows = []
    total_saved = 0
    skipped = 0

    pbar = tqdm(all_pairs, desc="Extracting", unit="video",
                bar_format="{l_bar}{bar:30}{r_bar}")

    for idx, (mp4, xml, src_type) in enumerate(pbar, 1):
        if src_type == "kisa":
            info = parse_kisa_xml(xml)
            prefix = "kisa"
        else:
            info = parse_aihub_xml(xml)
            prefix = "aihub"

        if info is None:
            pbar.write(f"[{idx:03d}/{total}] [{prefix}] {mp4.name}  ->  SKIP (XML parse failed)")
            continue

        # Skip if already extracted
        out_sub = out_dir / f"{prefix}_{mp4.stem}"
        if out_sub.exists() and any(out_sub.glob("*.jpg")):
            existing = len(list(out_sub.glob("*.jpg")))
            skipped += 1
            total_saved += existing
            pbar.write(f"[{idx:03d}/{total}] [{prefix:5s}] {mp4.name}  ->  SKIP ({existing} frames exist)")
            csv_rows.append({
                "source": prefix, "video": mp4.name,
                "starttime": fmt_time(info["starttime"]),
                "keyframe": info.get("keyframe", ""),
                "kp_x": info.get("kp_x", ""),
                "kp_y": info.get("kp_y", ""),
                "src_w": info["width"], "src_h": info["height"],
                "fps": info["fps"], "frames_saved": existing,
                "out_dir": str(out_sub),
            })
            continue

        out_sub.mkdir(exist_ok=True)
        n = extract_frames(mp4, info, out_sub, window_sec, sample_fps)
        total_saved += n
        st_str = fmt_time(info["starttime"])
        kp_str = f"KP=({info['kp_x']},{info['kp_y']})" if info.get("kp_x") is not None else "KP=N/A"
        pbar.write(f"[{idx:03d}/{total}] [{prefix:5s}] {mp4.name}  ST={st_str}  {kp_str}  -> {n} frames")
        pbar.set_postfix(saved=total_saved, skip=skipped)

        csv_rows.append({
            "source": prefix, "video": mp4.name,
            "starttime": st_str,
            "keyframe": info.get("keyframe", ""),
            "kp_x": info.get("kp_x", ""),
            "kp_y": info.get("kp_y", ""),
            "src_w": info["width"], "src_h": info["height"],
            "fps": info["fps"], "frames_saved": n,
            "out_dir": str(out_sub),
        })

    pbar.close()

    # Save index CSV
    csv_path = out_dir / "index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["source", "video", "starttime", "keyframe", "kp_x", "kp_y",
                      "src_w", "src_h", "fps", "frames_saved", "out_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    kisa_count = sum(1 for r in csv_rows if r["source"] == "kisa")
    aihub_count = sum(1 for r in csv_rows if r["source"] == "aihub")
    print(f"\n{'='*70}")
    print(f"  Done!  KISA: {kisa_count}  AI Hub: {aihub_count}  Total frames: {total_saved}  (skipped: {skipped})")
    print(f"  Index: {csv_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()