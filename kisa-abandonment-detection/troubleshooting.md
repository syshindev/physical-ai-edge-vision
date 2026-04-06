# Abandonment Detection — Troubleshooting

## Issue 1: MOG2 Noise from Trees and Shadows

**Problem**: MOG2 background subtraction detected swaying tree branches and moving shadows as foreground objects, generating dozens of false positive blobs per frame in outdoor scenes.

**Root Cause**: MOG2 learns a static background model during the 30-second warmup. Any persistent motion (wind-blown trees, sun-angle shadow shifts) that starts after warmup appears as new foreground. This is a fundamental characteristic of background subtraction — it cannot distinguish "real objects" from "visual changes."

**Solution**: Person path heatmap filtering. Record where confirmed persons have walked (150px radius circles on a heatmap), and only evaluate blobs whose center falls within the heatmap. Trees and shadows are almost never in areas where a person has walked.

**Result**: Eliminated >90% of MOG2 false positives without losing any real abandoned objects. See [algorithm-design.md](./algorithm-design.md#person-path-heatmap) for the heatmap design.

## Issue 2: Non-COCO Object Detection (Trash Bags, Packages)

**Problem**: KISA test videos include trash bags, wrapped packages, and generic bags as abandoned objects. YOLO (trained on COCO) cannot detect these — they are not in the 80 COCO classes.

**Root Cause**: YOLO is a class-specific detector. It can only find objects it was trained to recognize. "Trash bag on the ground" is not a COCO category, and fine-tuning for it would require a large labeled dataset that doesn't exist.

**Solution**: Used MOG2 background subtraction instead of YOLO for object detection. MOG2 is class-agnostic — it detects any region that differs from the background, regardless of what the object is. This made the system work for arbitrary object types.

**Result**: All object types (bags, trash bags, packages) detected without any class-specific training.

## Issue 3: Night + Snow MOG2 Noise

**Problem**: In night+snow videos (C00_146), MOG2 produced massive foreground regions covering most of the frame. Snow accumulation after background freeze was treated as new foreground everywhere.

**Root Cause**: After the 30-second warmup, the background model freezes. Snow falling and accumulating changes the visual appearance of every surface. MOG2 flags the entire snow-covered ground as "foreground" — a correct assessment from the algorithm's perspective, but useless for abandonment detection.

**Solution**: Complete MOG2 bypass in night mode (brightness < 100). DIFF frame comparison handles night detection independently. Night detection was already needed, so rather than trying to fix MOG2 for night scenes, the system simply doesn't use it.

**Result**: Night+snow videos (C00_146) went from FAIL to PASS. See [algorithm-design.md](./algorithm-design.md#daynight-mode-switching) for the mode switching design.

## Issue 4: Nighttime Person Flickering

**Problem**: At night, YOLO detected persons intermittently — 1 frame detected, 2 frames missed, 1 frame detected. This registered a person path at every flickering detection, contaminating the heatmap with false paths.

**Root Cause**: Low-light conditions produce noisy detections. The person is borderline visible, so YOLO's confidence fluctuates around the threshold frame-by-frame. Each detection above threshold registered a path point, even though no real person was consistently present.

**Solution**: Hysteresis filter requiring 30 consecutive YOLO detections (~1 second) before confirming a person. Additionally, minimum thresholds of 50 total frames and 5 seconds dwell time. Single-frame or brief detections are completely ignored.

**Result**: Nighttime person flickering no longer contaminates the heatmap. Only persons who are genuinely present for multiple seconds register paths. See [algorithm-design.md](./algorithm-design.md#person-confirmation) for hysteresis details.

## Issue 5: MOG2 and DIFF Timing Conflict

**Problem**: In videos where both MOG2 and DIFF detected the object, the system had to choose which detection time to report. MOG2 detected 1~5 seconds earlier than DIFF on average, but neither was consistently more accurate.

**Root Cause**: MOG2 detects statistical background deviation (fast but noisy). DIFF compares against a reference frame (slow but precise). Their detection times diverge because they measure different things — MOG2 reacts to the object appearing, DIFF reacts to sufficient visual contrast accumulating.

**Solution**: Weighted average when both detect with a gap > 5 seconds: `event_time = 0.6 * MOG2 + 0.4 * DIFF`. When gap <= 5s, use MOG2 directly. A 3-second wait period ensures both methods have time to report before committing.

**Result**: Timing accuracy improved on boundary cases (C00_036: +8s, C00_221: +9s — both within the +10s limit). See [algorithm-design.md](./algorithm-design.md#day-mode-weighted-average) for the averaging logic.

## Issue 6: Distant Small Object Miss (C00_036)

**Problem**: C00_036 is a distant/wide-angle shot where the abandoned trash bag appears very small in frame. Both MOG2 and DIFF initially missed it because the blob area was below minimum thresholds.

**Root Cause**: Default minimum areas (MOG2: 200px, DIFF: 150px) were tuned for typical CCTV distances. Distant shots produce objects smaller than these thresholds, causing them to be filtered out as noise.

**Solution**: Lowered minimum area thresholds — MOG2 to 150px, DIFF to 100px. Combined with person path filtering (which already removes most small noise blobs), the lower thresholds don't increase false positives.

**Result**: C00_036 detected successfully at +8s (within tolerance). The lower thresholds didn't cause regressions on other videos thanks to heatmap filtering.

## Issue 7: Bbox Size Inaccuracy

**Problem**: The reported bounding box for the abandoned object was often too small — only capturing part of the object, or offset from the actual object center.

**Root Cause**: MOG2 blobs don't correspond neatly to object boundaries. Partial foreground masks, noise artifacts, and gradual background adaptation cause the MOG2 blob to be smaller or shifted relative to the actual object.

**Solution**: After confirming an abandoned object, recalculate the bounding box using DIFF (reference frame vs current frame comparison). DIFF produces cleaner object boundaries because it compares full pixel values rather than statistical models.

**Result**: Bboxes accurately surround the abandoned object in the output visualization.

## Issue 8: Person Masking Covers Object

**Problem**: The person bbox mask (applied to MOG2 input to avoid detecting the person as foreground) also masked the abandoned object when the person was standing near it or placing it down.

**Root Cause**: The masking margin (originally 40px around person bbox) was too large. When the person bends down to place an object, their bbox extends to the object's location, and the margin covers the object entirely.

**Solution**: Reduced masking margin to 20px with an additional 5px downward extension. Also added a grace period: allow blob registration up to 10 seconds after the person leaves, so objects placed under the person mask can still be detected after the person walks away.

**Result**: Objects placed at the person's feet are detected within the grace period after the person moves away.

## Issue 9: MOG2 Early False Confirmation

**Problem**: Some noise blobs (tree shadows, camera shake) persisted long enough to pass the MOG2 static confirmation time (10 seconds), generating false events.

**Root Cause**: Blob confirmation only checked persistence duration. Long-lived noise (shadow at a consistent angle, permanent camera vibration) could satisfy the 10-second requirement.

**Solution**: Two additional filters:
1. **Stale blob filter**: Ignore blobs detected more than 30 seconds before the person left. These are background features, not abandoned objects.
2. **Weighted average cross-check**: Use the 3-second wait period and DIFF comparison to validate MOG2-only detections.

**Result**: False confirmations from persistent noise eliminated without affecting real object detection timing.

## Issue 10: MOG2 and DIFF Selection Conflict

**Problem**: When MOG2 detected first, the system immediately committed to that result. If DIFF later detected a more accurate time, it was ignored because the event was already confirmed.

**Root Cause**: The original "first-come-first-served" selection logic didn't account for the systematic timing difference between MOG2 (early) and DIFF (late but more accurate).

**Solution**: Added a 3-second wait period after the first detection. During this window, both MOG2 and DIFF results are collected. After the wait, the system applies the weighted average logic to produce the final event time.

**Result**: Both methods contribute to the final timing, producing more accurate event times than either alone.

## Issue 11: Night Mode MOG2 Selection Bug

**Problem**: In night videos, MOG2 would occasionally produce a blob before DIFF (despite being unreliable at night), and the system would commit to the MOG2 result before the night-mode DIFF-only logic could override it.

**Root Cause**: The night detection check ran after blob selection, not before. MOG2 blobs in night scenes could enter the confirmation pipeline before the night flag was evaluated.

**Solution**: Moved night detection check to the earliest stage — before any blob selection. In night mode, MOG2 results are discarded before they enter the pipeline, ensuring only DIFF results are processed.

**Result**: Night videos consistently use DIFF-only detection with no MOG2 interference.

## Issue 12: Person Must Leave Frame for SUSPECT

**Problem**: In some videos, the person places an object and walks to the far side of the frame but never actually exits. The original logic required person to leave the camera view to trigger SUSPECT, causing missed detections or very late timing.

**Root Cause**: The leave detection logic checked `person not in frame` as a binary condition. Scenarios where the person walks away but stays visible were not handled.

**Solution**: Added distance-based leave detection — SUSPECT triggers when the person moves 200px away from the object, regardless of whether they're still in frame. Event timing still uses frame-exit timestamp for accuracy.

**Result**: All videos detect correctly, including cases where the person remains partially visible. Visual transitions appear more natural.

## Issue 13: MOG2 Bbox Sudden Expansion

**Problem**: Occasionally, a confirmed object's bounding box would suddenly expand 3~5x in a single frame, engulfing a large area of the frame.

**Root Cause**: Two separate MOG2 noise blobs merged when an intermediate area became "foreground" (shadow shift connecting two regions). The merged blob's bounding box covered both original regions plus the gap between them.

**Solution**: Reject any frame where the blob area increases by more than 3x compared to the previous frame. The previous bbox is maintained until the blob stabilizes.

**Result**: Bbox remains stable through noise merging events. The 3x threshold is permissive enough to allow legitimate object growth (person placing additional items) while blocking noise merges.
