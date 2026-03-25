"""Collapse 10-video batch test — progress bar + result summary + file export"""
import os, sys
import cv2
from datetime import datetime
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from logger_config import logger
from collapse import CollapseMonitor

VIDEO_DIR = Path(r"C:\Users\gmission\Desktop\kisa_vid\falldown")
GT_DIR    = Path(r"C:\Users\gmission\Desktop\PerformanceEvaluation\gt\collapse")
MAP_DIR   = PROJECT_ROOT / "listmap"

KISA_LOWER = -2.0
KISA_UPPER = 10.0

VIDEOS = [
    # (video_file, map_file, gt_start_sec)
    ("C00_003_0003.mp4", "C00_003.map", 198),
    ("C00_023_0004.mp4", "C00_023.map", 197),
    ("C00_034_0005.mp4", "C00_034.map", 188),
    ("C00_060_0003.mp4", "C00_060.map", 211),
    ("C00_081_0002.mp4", "C00_081.map", 236),
    ("C00_146_0004.mp4", "C00_146.map", 112),
    ("C00_153_0005.mp4", "C00_153.map", 107),
    ("C00_188_0003.mp4", "C00_188.map", 128),
    ("C00_217_0004.mp4", "C00_217.map", 118),
    ("C00_235_0002.mp4", "C00_235.map", 112),
]


def time_str_to_sec(t: str) -> float:
    h, m, s = map(int, t.strip().split(":"))
    return h * 3600 + m * 60 + s


def sec_to_time_str(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_one(video_file, map_file, gt_start, monitor, log_path):
    video_path = VIDEO_DIR / video_file
    map_path   = MAP_DIR / map_file
    basename   = video_file.replace(".mp4", "")

    if not video_path.exists():
        return basename, "FILE_NOT_FOUND", gt_start, None, None
    if not map_path.exists():
        return basename, "MAP_NOT_FOUND", gt_start, None, None

    # Redirect logs to file
    logger.remove()
    log_id = logger.add(
        str(log_path),
        level="DEBUG",
        mode="w",
        encoding="utf-8",
        format="{time:HH:mm:ss.SSS} | {level:<8} | {message}",
        enqueue=False,
    )

    try:
        monitor.reset_for_new_video()
        if not monitor.parse_xml_zones(str(map_path)):
            return basename, "MAP_PARSE_FAIL", gt_start, None, None

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        BAR_W = 25

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 2 != 0:
                continue
            elapsed = frame_count / fps
            monitor.process_frame(frame, frame_count, fps, elapsed)

            if frame_count % 200 == 0 or frame_count == total_frames:
                pct  = frame_count / max(1, total_frames)
                fill = int(BAR_W * pct)
                bar  = '\u2588' * fill + '\u2591' * (BAR_W - fill)
                print(f"\r  [{bar}] {pct*100:.0f}%  ({frame_count}/{total_frames})", end="", flush=True)

        cap.release()

        event = getattr(monitor, "confirmed_fall_event", None)
        if event is None:
            return basename, "NoEvent", gt_start, None, None

        detected_sec = event["time"]
        detected_str = sec_to_time_str(detected_sec)
        diff = detected_sec - gt_start
        passed = KISA_LOWER <= diff <= KISA_UPPER

        return basename, detected_str, gt_start, diff, passed

    except Exception as e:
        logger.exception(f"Processing error: {e}")
        return basename, "ERROR", gt_start, None, None
    finally:
        logger.remove(log_id)


def main():
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / f"collapse_batch_{ts}"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    total = len(VIDEOS)

    print(f"\n{'='*70}")
    print(f"  KISA Collapse Batch Evaluation  |  {total} videos  |  {ts}")
    print(f"{'='*70}\n")

    monitor = CollapseMonitor()

    rows = []
    pass_n = fail_fn = fail_early = fail_late = skip_n = 0

    for idx, (video_file, map_file, gt_start) in enumerate(VIDEOS, 1):
        basename = video_file.replace(".mp4", "")
        gt_str   = sec_to_time_str(gt_start)
        log_path = log_dir / f"{basename}.log"

        print(f"[{idx:02d}/{total}] {basename}  GT={gt_str}({gt_start}s)  processing...", end="", flush=True)

        t_start = datetime.now()
        name, result, gt, diff, passed = run_one(video_file, map_file, gt_start, monitor, log_path)
        elapsed = (datetime.now() - t_start).total_seconds()

        if result in ("FILE_NOT_FOUND", "MAP_NOT_FOUND", "MAP_PARSE_FAIL"):
            print(f"\r[{idx:02d}/{total}] {basename}  ->  SKIP ({result})")
            rows.append((basename, gt_str, "---", "N/A", "SKIP"))
            skip_n += 1
        elif diff is not None:
            det_str  = result
            diff_txt = f"{diff:+.0f}s"
            if passed:
                status = "PASS"
                pass_n += 1
            elif diff < KISA_LOWER:
                status = "FAIL(early)"
                fail_early += 1
            else:
                status = "FAIL(late)"
                fail_late += 1
            mark = "\u2705" if passed else "\u274c"
            print(f"\r[{idx:02d}/{total}] {basename}  detected={det_str}  diff={diff_txt}  {mark}{status}  ({elapsed:.0f}s)")
            rows.append((basename, gt_str, det_str, diff_txt, status))
        else:
            fail_fn += 1
            print(f"\r[{idx:02d}/{total}] {basename}  detected=none  \u274cFAIL(missed)  ({elapsed:.0f}s)")
            rows.append((basename, gt_str, "none", "N/A", "FAIL(missed)"))

    # Summary table
    evaluated = total - skip_n
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"{'Video':<18} {'GT':>8} {'Detect':>8} {'Diff':>7}  Result")
    print(f"{'-'*70}")
    for name, gt, det, diff, status in rows:
        print(f"{name:<18} {gt:>8} {det:>8} {diff:>7}  {status}")
    print(sep)
    print(f"PASS  : {pass_n}/{evaluated}")
    print(f"FAIL  : {fail_fn+fail_early+fail_late}  (missed:{fail_fn}  early:{fail_early}  late:{fail_late})")
    if skip_n:
        print(f"SKIP  : {skip_n}")
    print(f"Accuracy: {pass_n/max(1,evaluated)*100:.1f}%  (KISA criteria -2s~+10s)")
    print(sep)

    # Save to file
    summary = out_dir / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"KISA Collapse Batch Evaluation  {ts}\n{sep}\n")
        f.write(f"{'Video':<18} {'GT':>8} {'Detect':>8} {'Diff':>7}  Result\n{'-'*70}\n")
        for name, gt, det, diff, status in rows:
            f.write(f"{name:<18} {gt:>8} {det:>8} {diff:>7}  {status}\n")
        f.write(f"{sep}\n")
        f.write(f"PASS  : {pass_n}/{evaluated}\n")
        f.write(f"FAIL  : {fail_fn+fail_early+fail_late}  (missed:{fail_fn}  early:{fail_early}  late:{fail_late})\n")
        f.write(f"Accuracy: {pass_n/max(1,evaluated)*100:.1f}%\n")

    print(f"\nLogs: {log_dir}")
    print(f"Summary: {summary}\n")


if __name__ == "__main__":
    main()
