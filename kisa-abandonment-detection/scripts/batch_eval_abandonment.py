"""
Batch evaluation for abandonment detection — run all test videos and produce pass/fail summary.

Processes each video through AbandonmentMonitor, compares detected event time
against ground truth, and generates a summary report.

Usage:
    python batch_eval_abandonment.py \
        --vid-dir /path/to/videos \
        --map-dir /path/to/map_files \
        --out-dir /path/to/results

    # With custom video list:
    python batch_eval_abandonment.py \
        --vid-dir /path/to/videos \
        --map-dir /path/to/map_files \
        --videos C00_015_0001:C00_015:216 C00_023_0001:C00_023:214

Video format: VIDEO_STEM:MAP_STEM:GT_START_SEC
    e.g., C00_015_0001:C00_015:216
"""

import argparse
import sys
import cv2
from datetime import datetime
from pathlib import Path


def seconds_to_time_str(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def time_str_to_sec(t: str) -> float:
    h, m, s = map(int, t.strip().split(":"))
    return h * 3600 + m * 60 + s


KISA_LOWER = -2.0
KISA_UPPER = 10.0

DEFAULT_VIDEOS = [
    ("C00_015_0001", "C00_015", 216),
    ("C00_023_0001", "C00_023", 214),
    ("C00_028_0001", "C00_028", 211),
    ("C00_036_0001", "C00_036", 209),
    ("C00_083_0001", "C00_083", 206),
    ("C00_146_0001", "C00_146", 118),
    ("C00_153_0001", "C00_153", 110),
    ("C00_188_0001", "C00_188", 120),
    ("C00_221_0001", "C00_221", 111),
    ("C00_230_0001", "C00_230", 110),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation for abandonment detection")
    parser.add_argument("--vid-dir", required=True, type=Path, help="Video directory")
    parser.add_argument("--map-dir", required=True, type=Path, help="Map file directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: results/abandonment_batch_TIMESTAMP)")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Video specs as STEM:MAP_STEM:GT_SEC (default: built-in list)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse video list
    if args.videos:
        videos = []
        for spec in args.videos:
            parts = spec.split(":")
            videos.append((parts[0], parts[1], int(parts[2])))
    else:
        videos = DEFAULT_VIDEOS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(f"results/abandonment_batch_{ts}")
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: In production, AbandonmentMonitor is imported from the project's src/ directory.
    # This script demonstrates the evaluation framework structure.
    # from abandonment import AbandonmentMonitor

    total = len(videos)
    print(f"\n{'='*70}")
    print(f"  Abandonment Batch Evaluation  |  {total} videos  |  {ts}")
    print(f"{'='*70}\n")

    rows = []
    pass_n = fail_n = 0

    for idx, (video_stem, map_stem, gt_start) in enumerate(videos, 1):
        video_path = args.vid_dir / f"{video_stem}.mp4"
        map_path = args.map_dir / f"{map_stem}.map"
        gt_str = seconds_to_time_str(gt_start)

        print(f"[{idx:02d}/{total}] {video_stem}  GT={gt_str}")

        if not video_path.exists():
            print(f"  -> SKIP (video not found)")
            rows.append((video_stem, gt_str, "---", "N/A", "SKIP"))
            continue

        # In production: monitor.reset_for_new_video(), process frames, check event
        # detected_sec = monitor.get_event_time()
        # diff = detected_sec - gt_start
        # passed = KISA_LOWER <= diff <= KISA_UPPER

        # Placeholder for demonstration
        rows.append((video_stem, gt_str, "---", "N/A", "NOT_RUN"))

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"{'Video':<18} {'GT':>8} {'Detected':>8} {'Diff':>7}  Result")
    print(f"{'-'*70}")
    for name, gt, det, diff, status in rows:
        print(f"{name:<18} {gt:>8} {det:>8} {diff:>7}  {status}")
    print(sep)

    # Save summary
    summary = out_dir / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"Abandonment Batch Evaluation  {ts}\n{sep}\n")
        for name, gt, det, diff, status in rows:
            f.write(f"{name:<18} {gt:>8} {det:>8} {diff:>7}  {status}\n")
        f.write(f"{sep}\n")

    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
