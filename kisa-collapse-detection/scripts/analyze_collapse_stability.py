"""Collapse batch log stability analysis — per-video visual stability metrics report"""
import re, sys
from pathlib import Path

def analyze_log(log_path: str) -> dict:
    """Extract stability metrics from log file"""
    stats = {
        "confirmed_count": 0,           # CONFIRMED count
        "confirmed_ids": [],            # list of CONFIRMED IDs
        "normal_reset_count": 0,        # NORMAL reset count after CONFIRMED
        "normal_reset_types": [],       # reset types (NONFALL/STAND/REAL-LOST)
        "pseudo_block_count": 0,        # pseudo ID block count
        "few_detect_block_count": 0,    # insufficient detection block count
        "fallback_hit_count": 0,        # FALLBACK HIT count
        "fallback_miss_count": 0,       # FALLBACK MISS count
        "gc_confirmed_count": 0,        # CONFIRMED tracks removed by GC
        "gc_suspect_count": 0,          # SUSPECT tracks removed by GC
        "kp_quality_reject": 0,         # keypoint quality rejection count
        "crop_none_count": 0,           # CROP-NONE count
        "rtmpose_retrack_count": 0,     # RTMPose re-detection success count
        "suspect_count": 0,             # SUSPECT entry count
        "suspect_to_normal": 0,         # SUSPECT to NORMAL count
        "is_dark": False,               # whether nighttime video
    }

    confirmed_ids_set = set()

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # CONFIRMED
            m = re.search(r"\[FIRST-CONFIRMED\].*ID:(-?\d+)", line)
            if m:
                stats["confirmed_count"] += 1
                stats["confirmed_ids"].append(int(m.group(1)))
                confirmed_ids_set.add(int(m.group(1)))

            # NORMAL reset
            if "[CONF->NORMAL by REAL-LOST]" in line:
                stats["normal_reset_count"] += 1
                stats["normal_reset_types"].append("REAL-LOST")
            if "[RESET->NORMAL by NONFALL]" in line:
                m2 = re.search(r"ID:(-?\d+)", line)
                if m2 and int(m2.group(1)) in confirmed_ids_set:
                    stats["normal_reset_count"] += 1
                    stats["normal_reset_types"].append("NONFALL(confirmed)")
                else:
                    stats["suspect_to_normal"] += 1
            if "[RESET->NORMAL by STAND]" in line:
                stats["normal_reset_count"] += 1
                stats["normal_reset_types"].append("STAND")

            # Blocking
            if "[CONFIRMED-BLOCK-PSEUDO]" in line:
                stats["pseudo_block_count"] += 1
            if "[CONFIRMED-BLOCK-FEW]" in line:
                stats["few_detect_block_count"] += 1

            # FALLBACK
            if "[FALLBACK-HIT]" in line:
                stats["fallback_hit_count"] += 1
            if "[FALLBACK-MISS]" in line:
                stats["fallback_miss_count"] += 1

            # GC
            m_gc = re.search(r"\[GC\] drop tid=(-?\d+) state=(\d+)", line)
            if m_gc:
                st = int(m_gc.group(2))
                if st == 2:  # CONFIRMED
                    stats["gc_confirmed_count"] += 1
                elif st == 1:  # SUSPECT
                    stats["gc_suspect_count"] += 1

            # Keypoint quality
            if "[KP-QUALITY-REJECT]" in line:
                stats["kp_quality_reject"] += 1

            # CROP-NONE
            if "[CROP-NONE]" in line:
                stats["crop_none_count"] += 1

            # RTMPose re-detection
            if "[RTMPOSE-RETRACK]" in line:
                stats["rtmpose_retrack_count"] += 1

            # SUSPECT
            if "fall suspect start" in line:
                stats["suspect_count"] += 1

            # Nighttime
            if "[DARK-CHANGE] dark=True" in line:
                stats["is_dark"] = True

    return stats


def stability_score(stats: dict) -> tuple:
    """Stability score (0-100) + issue list"""
    score = 100
    issues = []

    # No CONFIRMED (false negative)
    if stats["confirmed_count"] == 0:
        score -= 50
        issues.append("False negative: no CONFIRMED")

    # NORMAL reset after CONFIRMED
    if stats["normal_reset_count"] > 0:
        score -= stats["normal_reset_count"] * 15
        for rt in stats["normal_reset_types"]:
            issues.append(f"CONFIRMED→NORMAL ({rt})")

    # CONFIRMED removed by GC
    if stats["gc_confirmed_count"] > 0:
        score -= stats["gc_confirmed_count"] * 20
        issues.append(f"GC removed CONFIRMED {stats['gc_confirmed_count']} time(s)")

    # Excessive pseudo ID blocks (nighttime noise)
    if stats["pseudo_block_count"] > 5:
        score -= 5
        issues.append(f"pseudo block {stats['pseudo_block_count']} time(s) (nighttime noise)")

    # Excessive CROP-NONE
    if stats["crop_none_count"] > 10:
        score -= 5
        issues.append(f"CROP-NONE {stats['crop_none_count']} time(s) (small bbox)")

    # High FALLBACK dependency
    fb_total = stats["fallback_hit_count"] + stats["fallback_miss_count"]
    if fb_total > 20:
        score -= 5
        issues.append(f"High FALLBACK dependency ({fb_total} time(s))")

    score = max(0, score)
    return score, issues


def main():
    # Find the most recent batch result
    results_dir = Path(__file__).parent.parent / "results"
    collapse_dirs = sorted(
        [d for d in results_dir.iterdir() if d.name.startswith("collapse_batch_")],
        reverse=True
    )

    if not collapse_dirs:
        print("No batch results found. Run batch_eval_collapse.py first.")
        return

    batch_dir = collapse_dirs[0]
    log_dir = batch_dir / "logs"

    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  Collapse Stability Analysis  |  {batch_dir.name}")
    print(f"{'='*70}\n")

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        print("No log files found")
        return

    all_stats = []
    for lf in log_files:
        stats = analyze_log(str(lf))
        score, issues = stability_score(stats)
        all_stats.append((lf.stem, stats, score, issues))

    # Print report
    print(f"{'Video':<20} {'Score':>4}  {'Night':>4}  {'Conf':>4}  {'Reset':>4}  {'GC-C':>4}  {'FB':>4}  {'RTM':>4}  Issues")
    print("-" * 100)

    for name, stats, score, issues in all_stats:
        dark = "Y" if stats["is_dark"] else "N"
        fb = stats["fallback_hit_count"] + stats["fallback_miss_count"]
        issue_str = " | ".join(issues) if issues else "OK"
        color_score = f"{score:3d}"

        print(f"{name:<20} {color_score:>4}  {dark:>4}  {stats['confirmed_count']:>4}  "
              f"{stats['normal_reset_count']:>4}  {stats['gc_confirmed_count']:>4}  "
              f"{fb:>4}  {stats['rtmpose_retrack_count']:>4}  {issue_str}")

    print("-" * 100)

    # Overall summary
    avg_score = sum(s for _, _, s, _ in all_stats) / len(all_stats)
    problem_videos = [(n, s, i) for n, _, s, i in all_stats if s < 80]

    print(f"\nAverage stability score: {avg_score:.0f}/100")
    if problem_videos:
        print(f"\nProblem videos ({len(problem_videos)}):")
        for name, score, issues in problem_videos:
            print(f"  - {name} ({score} pts): {' | '.join(issues)}")
    else:
        print("All videos stable (80 pts or above)")

    # Save to file
    report_path = batch_dir / "stability_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Collapse Stability Analysis  {batch_dir.name}\n")
        f.write(f"{'='*70}\n\n")
        for name, stats, score, issues in all_stats:
            f.write(f"{name}: {score} pts\n")
            f.write(f"  night={stats['is_dark']}, confirmed={stats['confirmed_count']}, "
                    f"reset={stats['normal_reset_count']}, GC-C={stats['gc_confirmed_count']}\n")
            f.write(f"  FB-HIT={stats['fallback_hit_count']}, FB-MISS={stats['fallback_miss_count']}, "
                    f"RTM-RETRACK={stats['rtmpose_retrack_count']}\n")
            f.write(f"  CROP-NONE={stats['crop_none_count']}, KP-REJECT={stats['kp_quality_reject']}\n")
            f.write(f"  PSEUDO-BLOCK={stats['pseudo_block_count']}, FEW-BLOCK={stats['few_detect_block_count']}\n")
            if issues:
                f.write(f"  Issues: {' | '.join(issues)}\n")
            f.write("\n")
        f.write(f"Average stability score: {avg_score:.0f}/100\n")

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
