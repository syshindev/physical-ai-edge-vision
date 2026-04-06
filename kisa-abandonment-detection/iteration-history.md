# Abandonment Detection — Iteration History

This module required the most exploration of all KISA detection systems — 10 distinct approaches were attempted before reaching 10/10 PASS. This document records what was tried, why it failed, and what was ultimately kept.

## Summary

| # | Approach | Core Idea | Result | Outcome |
|---|----------|-----------|--------|---------|
| 1 | MOG2 background subtraction | Detect foreground objects after background learning | 3/10 | Tree/shadow noise |
| 2 | YOLO object detection | Detect objects directly using COCO classes | Fail | Floor-level objects undetectable |
| 3 | YOLO belongings tracking | Track objects person is carrying | Partial | Non-COCO objects (trash bags) missed |
| 4 | MOG2 + person path filter | Only check blobs where person walked | 3/10 | Person masking covers object |
| 5 | Day MOG2 / Night belongings | Hybrid day/night strategy | 6/10 | YOLO can't detect objects at night |
| 6 | Before/After frame comparison | Compare early frame vs current | 4/10 | Misses objects MOG2 catches |
| 7 | MOG2 + DIFF dual | Either method triggers detection | 6→8/10 | Selection logic limitations |
| 8 | Real-time DIFF | Per-frame reference comparison | 1/10 | Person shadow/foot masking leaks |
| 9 | MOG2 + DIFF + hysteresis + day/night | Night DIFF-only, day weighted average | **10/10** | Boundary cases tight (1~2s margin) |
| 10 | #9 + distance-based leave detection | SUSPECT at 200px distance from object | **10/10** | Final version |

**Final state**: v10.1 — approach #9 core logic + distance-based leave trigger from #10 + original frame-exit timing.

---

## Approach 1: MOG2 Background Subtraction

### Idea
Use OpenCV MOG2 to learn the background, then detect any new foreground object as potentially abandoned.

### Problem
MOG2 treats everything that differs from the learned background as foreground:
- Tree branches swaying in wind → large foreground blobs
- Shadows moving with sun angle → phantom objects
- Camera auto-exposure adjustment → entire frame flickers

### Result: 3/10 PASS
Most videos had too many false positive blobs to isolate the actual abandoned object.

---

## Approach 2: YOLO Object Detection

### Idea
Use YOLO to directly detect the abandoned object (backpack, suitcase, etc.) by COCO class.

### Problem
KISA test objects include trash bags, generic bags, and wrapped items — none are COCO classes. Even for COCO objects like "backpack," detection fails when the object is on the ground (YOLO is trained on objects at typical heights, not floor-level).

### Result: Failed
Fundamentally incompatible with the problem — KISA objects are deliberately non-standard.

---

## Approach 3: YOLO Belongings Tracking

### Idea
Track objects the person is carrying (detected by YOLO while held), then flag when the object appears on the ground without the person nearby.

### Problem
Only works for COCO-class objects. Trash bags, wrapped packages, and other non-standard items cannot be tracked. Even trackable objects are often occluded while being carried.

### Result: Partial success
Worked for backpacks/suitcases in 3~4 videos, but completely failed on non-COCO objects.

---

## Approach 4: MOG2 + Person Path Filter

### Idea
Combine approach #1 with person tracking — only consider MOG2 blobs in areas where a person has walked (heatmap filtering).

### Problem
Person masking (removing person bbox from MOG2 input to avoid detecting the person as foreground) also masks the object if the person is standing near it. The 20px margin around the person bbox covers small objects at the person's feet.

### Result: 3/10 PASS
Path filtering removed tree/shadow noise, but masking interference created a new failure mode.

---

## Approach 5: Day MOG2 / Night Belongings Hybrid

### Idea
Use MOG2 during daytime (where it works reasonably) and YOLO belongings tracking at night (where MOG2 fails).

### Problem
YOLO can't detect objects at night either — low contrast and sensor noise make object detection unreliable in dark conditions.

### Result: 6/10 PASS
Daytime improved, but night videos all failed.

---

## Approach 6: Before/After Frame Comparison (DIFF)

### Idea
Save an early frame as reference, compare against current frame to find newly appeared objects.

### Problem
DIFF alone misses objects that MOG2 catches — particularly objects with similar color to the background. DIFF requires a minimum contrast difference to trigger, while MOG2's statistical model can detect subtler changes.

### Result: 4/10 PASS
Complementary to MOG2 but not sufficient alone.

---

## Approach 7: MOG2 + DIFF Dual Detection

### Idea
Run both methods; if either detects an object, trigger the event.

### Problem
"Either-or" selection logic was too aggressive — MOG2 false positives in daytime weren't filtered, and the timing of which method to trust was unreliable.

### Result: 6/10 → 8/10 PASS
Iterative tuning of thresholds and areas improved from 6 to 8, but two nighttime videos remained stuck due to MOG2 noise.

---

## Approach 8: Real-Time DIFF

### Idea
Instead of comparing against a fixed reference frame, compare every frame against the previous frame to detect newly static objects.

### Problem
Person shadows and foot positions create persistent diff regions that don't clear after the person walks away. The masking was never clean enough — residual diff from the person overwhelmed the actual object signal.

### Result: 1/10 PASS
Catastrophic regression. Per-frame comparison is too noisy for this use case.

---

## Approach 9: MOG2 + DIFF + Hysteresis + Day/Night Split

### Idea
Combine the best elements from previous attempts:
- MOG2 + DIFF dual detection (from #7)
- Person path heatmap filtering (from #4, improved masking)
- Hysteresis person confirmation (new — 30 consecutive frames)
- Day/night automatic switching (new — brightness threshold)
- Weighted average timing (new — 0.6 MOG2 + 0.4 DIFF)

### Key Changes from Previous Attempts
1. **Night: DIFF only** — Completely bypass MOG2 at night instead of trying to filter its noise
2. **Hysteresis** — 30-frame consecutive detection before confirming person (blocks night flickering)
3. **Weighted average** — Don't just pick MOG2 or DIFF; average when both detect
4. **3-second wait** — Wait for both methods before committing to a detection time

### Result: 10/10 PASS (v9.4)
First time reaching full pass. Stability verified with 2 consecutive runs producing identical results.

---

## Approach 10: Distance-Based Leave Detection

### Idea
Instead of requiring the person to leave the camera frame entirely, trigger SUSPECT when they move 200px away from the object.

### v10.0: Changed Both Trigger AND Timing
Used distance-based timing for event calculation.

### Result: 8/10 (regression)
Distance-based timing was less accurate than frame-exit timing.

### v10.1: Distance Trigger + Original Timing
Keep distance-based SUSPECT trigger (visual improvement) but revert to frame-exit timing for event calculation.

### Result: 10/10 PASS (final)
Best of both worlds — natural visual transitions with accurate timing.

---

## Detailed Tuning History

| Version | Changes | Result |
|---------|---------|--------|
| v7.0 | Dual detection, min selection | 6/10 |
| v7.1 | diff_thr 30, path_radius 120 | 7/10 |
| v7.3 | min_person 5s + 50 frames | 7/10 |
| v7.4 | diff_thr 25, path_radius 150, blob filter | 8/10 |
| v7.5 | diff_thr 20, area 100/150, blur 5x5 | 8/10 |
| v9.0 | + hysteresis 30 frames | 8/10 |
| v9.2 | + day/night split (brightness 100) | 8/10 (night bug) |
| v9.3 | Night DIFF-only + day weighted average | 8/10 |
| v9.4 | + 3-second wait logic | **10/10** |
| v9.5 | Stability verification (2 runs) | **10/10** |
| v10.0 | + distance-based (timing also changed) | 8/10 (regression) |
| v10.1 | Distance trigger + original timing | **10/10 (final)** |

### Key Observations

- **v7.x → v9.0**: Most progress came from lowering thresholds and adding filters. But nighttime videos remained stuck at 8/10 regardless of threshold tuning.
- **v9.0 → v9.4**: The breakthrough was architectural — separating day/night modes and waiting for both methods before committing. This solved the last 2 nighttime videos.
- **v10.0 → v10.1**: Lesson that visual improvements (distance-based SUSPECT) must not change timing logic. Decoupling trigger and timing was the correct design.

---

## Legacy Code (Unused)

The inherited codebase (`from_Gayrat/abandonment.py`) used YOWO v2 action recognition:
- Model: `yowo_v2_nano_ava.onnx` (weights file missing — could not run)
- Approach: AVA dataset "throw/put down" action detection
- Problem: AVA actions don't match KISA abandonment scenario (person placing item ≠ "throwing")
- Decision: Abandoned entirely, built from scratch
