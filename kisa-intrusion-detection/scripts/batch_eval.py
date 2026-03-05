"""
Batch evaluation script for KISA intrusion detection.

Runs all test videos through the intrusion detection pipeline,
compares detected event times against ground truth XML files,
and produces a pass/fail summary report.

Usage:
    python batch_eval.py \
        --video-dir /path/to/videos \
        --gt-dir /path/to/ground-truth \
        --map-dir /path/to/map-files \
        --config /path/to/config.xml

Output:
    results/batch_YYYYMMDD_HHMMSS/
    ├── summary.txt    (pass/fail table)
    └── logs/          (per-video debug logs)
"""

import argparse
import sys
import os
import cv2
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from loguru import logger

KISA_LOWER = -2.0   # GT tolerance lower bound (seconds)
KISA_UPPER = 10.0   # GT tolerance upper bound (seconds)


def parse_args():
    parser = argparse.ArgumentParser(description="KISA intrusion batch evaluation")
    parser.add_argument("--video-dir", required=True, help="Directory containing test videos (.mp4)")
    parser.add_argument("--gt-dir", required=True, help="Directory containing ground truth XMLs")
    parser.add_argument("--map-dir", required=True, help="Directory containing .map files")
    parser.add_argument("--config", required=True, help="Path to config.xml with video list")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: results/batch_<timestamp>)")
    return parser.parse_args()


def time_str_to_sec(t: str) -> float:
    """Convert 'HH:MM:SS' to float seconds."""
    h, m, s = map(int, t.strip().split(":"))
    return h * 3600 + m * 60 + s


def get_video_list(config_xml: Path):
    """Parse config.xml for the list of video filenames."""
    tree = ET.parse(config_xml)
    root = tree.getroot()
    return [f.find("Name").text.strip() for f in root.findall(".//File")]


def get_gt_start(gt_path: Path):
    """Extract StartTime string from a ground truth XML. Returns None if missing."""
    if not gt_path.exists():
        return None
    tree = ET.parse(gt_path)
    alarm = tree.getroot().find(".//Alarm")
    if alarm is None:
        return None
    st = alarm.find("StartTime")
    return st.text.strip() if st is not None else None


def get_map_path(video_name: str, map_dir: Path) -> Path:
    """Derive .map file path from video filename. e.g., C00_005_0001.mp4 -> C00_005.map"""
    base = "_".join(video_name.replace(".mp4", "").split("_")[:2])
    return map_dir / f"{base}.map"


def run_single(video_path: Path, map_path: Path, log_path: Path):
    """
    Process a single video through IntrusionMonitor.
    Returns the list of intrusion events, or None on failure.
    """
    # NOTE: IntrusionMonitor is the core detection module (not included in this repo).
    # This function demonstrates the evaluation interface.
    from intrusion import IntrusionMonitor

    # Replace default loguru handler with a file handler for this video
    logger.remove()
    log_id = logger.add(
        str(log_path),
        level="DEBUG",
        mode="w",
        encoding="utf-8",
        format="{time:HH:mm:ss.SSS} | {level:<8} | {message}",
        enqueue=False,
    )

    events = None
    try:
        monitor = IntrusionMonitor()
        if not monitor.parse_xml_zones(str(map_path)):
            logger.error(f"Zone parsing failed: {map_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        BAR_W = 25

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            elapsed = frame_count / fps
            monitor.process_frame(frame, frame_count, fps, elapsed)

            if frame_count % 200 == 0 or frame_count == total_frames:
                pct = frame_count / max(1, total_frames)
                fill = int(BAR_W * pct)
                bar = "\u2588" * fill + "\u2591" * (BAR_W - fill)
                print(f"\r  [{bar}] {pct*100:.0f}%  ({frame_count}/{total_frames})", end="", flush=True)

        cap.release()
        monitor.finalize_events(frame_count / fps)
        events = monitor.intrusion_events[:]

    except Exception as e:
        logger.exception(f"Processing error: {e}")
        events = None
    finally:
        logger.remove(log_id)

    return events


def main():
    args = parse_args()

    video_dir = Path(args.video_dir)
    gt_dir = Path(args.gt_dir)
    map_dir = Path(args.map_dir)
    config_xml = Path(args.config)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"batch_{ts}"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    videos = get_video_list(config_xml)
    total = len(videos)

    print(f"\n{'='*70}")
    print(f"  KISA Intrusion Batch Evaluation  |  {total} videos  |  {ts}")
    print(f"{'='*70}\n")

    rows = []
    pass_n = fail_fn = fail_early = fail_late = skip_n = 0

    for idx, vname in enumerate(videos, 1):
        vpath = video_dir / vname
        gtpath = gt_dir / vname.replace(".mp4", ".xml")
        mpath = get_map_path(vname, map_dir)
        lpath = log_dir / vname.replace(".mp4", ".log")

        gt_str = get_gt_start(gtpath)

        # Check required files
        missing = []
        if not vpath.exists():
            missing.append("video")
        if not mpath.exists():
            missing.append("map")
        if gt_str is None:
            missing.append("GT")
        if missing:
            print(f"[{idx:02d}/{total}] {vname}  ->  SKIP ({', '.join(missing)} missing)")
            rows.append((vname, "?", "SKIP", "N/A", "SKIP"))
            skip_n += 1
            continue

        gt_sec = time_str_to_sec(gt_str)
        print(f"[{idx:02d}/{total}] {vname}  GT={gt_str}({gt_sec:.0f}s)  processing...", end="", flush=True)

        t_start = datetime.now()
        events = run_single(vpath, mpath, lpath)
        elapsed = (datetime.now() - t_start).total_seconds()

        if not events:
            diff_s = None
            det_str = "none"
            status = "FAIL(FN)"
            fail_fn += 1
        else:
            det_str = events[0]["start_time"]
            det_sec = time_str_to_sec(det_str)
            diff_s = det_sec - gt_sec
            if KISA_LOWER <= diff_s <= KISA_UPPER:
                status = "PASS"
                pass_n += 1
            elif diff_s < KISA_LOWER:
                status = "FAIL(early)"
                fail_early += 1
            else:
                status = "FAIL(late)"
                fail_late += 1

        diff_txt = f"{diff_s:+.0f}s" if diff_s is not None else "N/A"
        mark = "PASS" if status == "PASS" else "FAIL"
        print(f"  det={det_str}  diff={diff_txt}  {mark}  ({elapsed:.0f}s)")
        rows.append((vname, gt_str, det_str, diff_txt, status))

    evaluated = total - skip_n
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"{'Video':<22} {'GT':>8} {'Det':>8} {'Diff':>7}  Result")
    print(f"{'-'*70}")
    for vname, gt, det, diff, status in rows:
        print(f"{vname:<22} {gt:>8} {det:>8} {diff:>7}  {status}")
    print(sep)
    print(f"PASS  : {pass_n}/{evaluated}")
    print(f"FAIL  : {fail_fn+fail_early+fail_late}  (FN:{fail_fn}  early:{fail_early}  late:{fail_late})")
    if skip_n:
        print(f"SKIP  : {skip_n}")
    print(f"Accuracy: {pass_n/max(1,evaluated)*100:.1f}%  (KISA tolerance: {KISA_LOWER}s ~ +{KISA_UPPER}s)")
    print(sep)

    # Write summary file
    summary = out_dir / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"KISA Intrusion Batch Evaluation  {ts}\n{sep}\n")
        f.write(f"{'Video':<22} {'GT':>8} {'Det':>8} {'Diff':>7}  Result\n{'-'*70}\n")
        for vname, gt, det, diff, status in rows:
            f.write(f"{vname:<22} {gt:>8} {det:>8} {diff:>7}  {status}\n")
        f.write(f"{sep}\n")
        f.write(f"PASS  : {pass_n}/{evaluated}\n")
        f.write(f"FAIL  : {fail_fn+fail_early+fail_late}  (FN:{fail_fn}  early:{fail_early}  late:{fail_late})\n")
        f.write(f"Accuracy: {pass_n/max(1,evaluated)*100:.1f}%\n")

    print(f"\nLogs: {log_dir}")
    print(f"Summary: {summary}\n")


if __name__ == "__main__":
    main()