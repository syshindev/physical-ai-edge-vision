# Collapse Detection — Troubleshooting

## Issue 1: XCLIP Score Flickering Breaks Fall Counter

**Problem**: A person clearly lying on the ground would get XCLIP scores like [0.85, 0.72, 0.91, 0.65, 0.88] across consecutive frames. The legacy system used raw scores — if even one frame dropped below the threshold, the entire 5-second confirmation counter reset to zero.

**Root Cause**: XCLIP is a zero-shot classifier operating on short video clips. Its output naturally fluctuates because small changes in the 8-frame input window (lighting shift, slight camera motion, crop boundary) cause different probability distributions. The legacy code treated each classification as independent and binary.

**Solution**: Replaced raw score comparison with EMA (Exponential Moving Average) scoring:
```
ema_new = 0.75 * ema_prev + 0.25 * current_score
```
Additionally, SUSPECT reset now requires `non_fall_count >= 3` consecutive non-fall frames instead of a single bad frame. This means one or two flickering frames can't destroy accumulated evidence.

**Result**: Stable score tracking through natural XCLIP fluctuation. Falls that previously took 15+ seconds to confirm (due to repeated resets) now confirm reliably at 5 seconds. See [algorithm-design.md](./algorithm-design.md#ema-scoring) for the full EMA design.

## Issue 2: Tracker ID Loss During Occlusion

**Problem**: When a person was briefly occluded (behind a pole, another person passing), BoT-SORT would lose the track and assign a new ID when the person reappeared. All accumulated fall evidence (EMA score, suspect timer, crop history) was tied to the old ID and effectively lost.

**Root Cause**: BoT-SORT's re-identification relies on appearance features and motion prediction. Brief occlusions break both — the person disappears from view and their predicted position drifts. When they reappear, the tracker sees them as a "new" person.

**Solution**: Three-layer tracking recovery system:

1. **ID Stitching**: When a new track appears, check if it matches a recently lost track by IOU (>= 0.30) or proximity (<= 180px). If matched, inherit all state from the old track.
2. **Fallback Detection**: For SUSPECT/CONFIRMED tracks that lost primary tracking, crop the last known region and run `model.predict()` at lower confidence to re-detect.
3. **Keepalive**: When even fallback detection fails, maintain the last known bounding box — SUSPECT for up to 6s, CONFIRMED for up to 30s. EMA decays slowly (0.999) during keepalive.

**Result**: Falls are tracked continuously through brief occlusions. Before this fix, occlusion during the 5-second SUSPECT window would always cause a missed detection. See [algorithm-design.md](./algorithm-design.md#tracking-recovery) for the full recovery design.

## Issue 3: Night Scene Systematic Failures

**Problem**: Nighttime CCTV footage produced near-zero detection rates. Persons were either not detected by YOLO, or their crops were too dark for XCLIP to classify correctly.

**Root Cause**: Multiple compounding factors:
- Low contrast: person and background pixel values overlap
- Higher noise: sensor gain increases in low light
- Smaller apparent sizes: dark clothing blends with dark surroundings, making effective bbox smaller
- XCLIP trained on well-lit data: dark crops produce unreliable probability distributions

**Solution**: 5-stage adaptive night mode pipeline:

| Stage | Day | Night | Why |
|-------|-----|-------|-----|
| 1. Preprocessing | None | CLAHE + brightness(1.15) + blur | Enhance contrast before detection |
| 2. Detection conf | 0.10 | 0.05–0.06 | Lower threshold to catch dim persons |
| 3. Resolution | 960px | 1280px | More pixels for small/dark persons |
| 4. Entry threshold | EMA >= 0.70 | EMA >= 0.62 | Accept weaker XCLIP signal in dark |
| 5. Confirm gates | AR >= 1.35 | AR >= 1.10 | Relax shape requirements |

Night detection uses per-frame HSV + grayscale analysis:
```
is_dark = (v_median < 105) or (v_p10 < 60) or (gray_mean < 95) or (gray_p10 < 55)
```

**Result**: Night scene detection rate went from near-zero to matching daytime performance in pre-test evaluation. See [algorithm-design.md](./algorithm-design.md#night-mode-pipeline) for the full pipeline design.

## Issue 4: False Positives from Bending and Sitting

**Problem**: XCLIP classified bending over (picking something up), sitting down on a bench, or crouching as "falling" with high confidence (0.70+). These poses share visual similarity with actual falls — the person's silhouette becomes horizontal.

**Root Cause**: XCLIP's zero-shot classification compares visual similarity to text prompts. "A person lying on the ground" and "a person bending forward" produce similar video features because both involve a person transitioning from vertical to horizontal orientation.

**Solution**: Multi-evidence requirement for SUSPECT entry. High XCLIP score alone is not sufficient — at least one physical evidence must also be present:

- **Y-velocity spike** (fall_motion_count >= 3): Real falls have sudden downward center-of-mass motion; bending is gradual
- **Bbox height drop >= 30%**: Person going from standing to lying causes a dramatic height reduction in one frame
- **Horizontal aspect ratio >= 1.20**: Lying person has width > height
- **XCLIP transition labels + aspect ratio**: "Collapsing" or "falling forward/backward" labels combined with horizontal bbox

Additionally, an **area jump filter** blocks entry when bbox area increases by >= 160% with aspect ratio > 1.6, which indicates a detection error (two people merging) rather than a real fall.

**Result**: Eliminated false positives from daily activities (bending, sitting, crouching) while maintaining sensitivity to real falls. See [algorithm-design.md](./algorithm-design.md#multi-evidence-system) for the full evidence requirements.

## Issue 5: PTS Timestamp Jumps Corrupting State Timers

**Problem**: The SUSPECT timer (which counts to 5 seconds) would sometimes jump from 2s directly to 8s, causing immediate CONFIRMED transition without actually accumulating enough evidence. Other times, the timer appeared to go backward.

**Root Cause**: Video streams (especially RTSP or re-encoded files) can have non-monotonic PTS (Presentation Timestamp) values. Seek operations, stream restarts, or container format issues cause the elapsed time to jump by several seconds. The legacy code used raw elapsed_time directly in all calculations.

**Solution**: PTS jump correction that maintains a corrected internal clock:
```python
dt_raw = raw - self._last_raw_t
if dt_raw > 2.0:    # Jump detected
    dt_use = 1.0 / max(1.0, fps)  # Use single frame duration
elif dt_raw < -0.5:  # Backward jump
    dt_use = 0.0
else:
    dt_use = dt_raw
self._corr_t += dt_use
```

All internal timers (suspect duration, confirmed hold, stand exit, etc.) use `_corr_t` instead of raw PTS.

**Result**: State transitions are immune to PTS discontinuities. The 5-second SUSPECT window always represents 5 actual seconds of evidence. See [algorithm-design.md](./algorithm-design.md#pts-jump-correction) for details.

## Issue 6: Fallback Detection Box Jumping

**Problem**: When fallback detection (re-detection in cropped ROI) found the person, the bounding box would sometimes jump dramatically between frames — shifting by 50+ pixels or changing size by 30%+. This caused the visualization to flicker and the velocity/aspect ratio calculations to produce false evidence.

**Root Cause**: Fallback detection runs at lower confidence on a cropped region. The detection is noisier than primary tracking because:
- Lower confidence threshold admits more uncertain detections
- ROI cropping changes the spatial context the model sees
- No temporal smoothing from the tracker (BoT-SORT not involved)

**Solution**: Two-stage filtering for fallback detections:

1. **Distance + area filter**: Reject detections too far from previous box (> 120–260px depending on state) or with area ratio outside [0.45, 2.2]
2. **Box EMA smoothing**: Apply exponential moving average to box coordinates (alpha=0.65 for CONFIRMED, 0.70 for others) to prevent jumps

```python
box_ema[tid] = alpha * box_ema[tid] + (1 - alpha) * new_box
```

**Result**: Smooth bounding box transitions during fallback, preventing false velocity spikes and aspect ratio changes from triggering incorrect state transitions.

## Issue 7: Warmup Period False Detections

**Problem**: In the first few seconds of video processing, YOLO would sometimes produce spurious detections (partial persons at frame edges, auto-exposure artifacts) that triggered SUSPECT state before the camera and model had stabilized.

**Root Cause**: Many CCTV cameras perform auto-exposure and white balance adjustment in the first 2–5 seconds. The resulting brightness/contrast changes create temporary artifacts that look like person silhouettes to the detector. Additionally, the first few XCLIP classifications lack temporal context (sequence not yet full).

**Solution**: 5-second warmup period with conservative behavior:
- Skip median/Gaussian blur during warmup (avoid spreading noise)
- Increase minimum bbox size requirements in dark conditions
- XCLIP sequence needs 8 frames × 0.15s interval = 1.2s minimum before first classification
- Combined effect: no false triggers in the first 5 seconds

**Result**: Eliminated startup false positives without delaying detection of actual falls that happen after the warmup window.

## Issue 8: XCLIP Crop Too Small for Reliable Classification

**Problem**: Small or distant persons produced tiny crops (< 30×60 pixels) that, when resized to XCLIP's required 224×224 input, became extremely blurry. The resulting classifications were essentially random noise.

**Root Cause**: XCLIP was designed for reasonably-sized video clips. When a 25×50 pixel crop is upscaled 8x to 224×224, the model receives a blocky, featureless image that doesn't match its training distribution.

**Solution**: State-dependent minimum crop sizes:

| State | Min Width | Min Height | Reasoning |
|-------|-----------|------------|-----------|
| NORMAL | 30px | 60px | Standard — reject unreliable crops |
| SUSPECT/CONFIRMED | 12px | 20px | Relaxed — don't lose track of fallen person |
| Dark + NORMAL | 18px | 40px | Slightly relaxed for night scenes |

When a crop is rejected, the XCLIP result is `None` and the EMA decays at its state-dependent rate rather than being updated with noise. This preserves the existing score through brief periods of unreliable crops.

**Result**: XCLIP only processes crops where its classification is meaningful, preventing noise injection into the EMA score.