# Abandonment Detection — Algorithm Design

## Detection Scenario

A person places an object (bag, trash bag, etc.) on the ground and leaves. The object remains unattended. An event is generated when the object has been abandoned for 10 seconds.

- **KISA event criterion**: Object placed on ground + 10 seconds elapsed
- **Detection tolerance**: -2s ~ +10s

## Dual Detection Pipeline

The system runs two independent object detection methods in parallel:

### MOG2 Background Subtraction

MOG2 (Mixture of Gaussians v2) maintains a statistical background model and identifies foreground objects — regions that differ from the learned background.

**Strengths**: Detects any stationary object regardless of type (bags, trash bags, boxes — none need to be in COCO classes).

**Weaknesses**: Produces noise from trees, shadows, snow, and lighting changes. After background freeze (30s warmup), any new persistent change becomes "foreground."

**Parameters**:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `bg_warmup_sec` | 30s | Background learning period before detection starts |
| `min_object_area_mog` | 150px | Minimum blob area to register |
| `mog_static_confirm_sec` | 10s | Blob must persist for 10s to confirm |
| `person_margin_px` | 20px | Mask margin around person bboxes |

### DIFF Frame Comparison

Compares a reference frame (captured early in the video) against the current frame to find newly appeared objects.

**Strengths**: Catches low-contrast objects that MOG2 misses (dark bags on dark ground). Less affected by gradual lighting changes.

**Weaknesses**: Slower to detect — needs multiple frames of confirmation. Sensitive to person shadows and residual motion.

**Parameters**:
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `diff_threshold` | 20 | Pixel intensity difference to flag as changed |
| `diff_confirm_frames` | 3 | Consecutive frames required to confirm blob |
| `min_object_area_diff` | 100px | Minimum blob area |
| `GaussianBlur` | (5, 5) | Blur kernel to reduce noise before comparison |

## Day/Night Mode Switching

The system measures frame brightness and switches detection strategy automatically:

```
brightness = mean(grayscale_frame)

if brightness >= 100:   # Day mode
    -> MOG2 primary, DIFF supplementary
    -> weighted average when both detect
else:                   # Night mode
    -> DIFF only
    -> MOG2 results completely ignored
```

### Why Night Bypasses MOG2

After the 30-second warmup, MOG2 freezes the background model. In night+snow conditions:
- Snow accumulation after freeze = new foreground everywhere
- Low-light sensor noise creates phantom blobs
- Tree branches moving in wind register as large foreground regions

These are not fixable with threshold tuning — they are fundamental to how background subtraction works in non-static scenes. Complete MOG2 bypass at night is simpler and more reliable than trying to filter the noise.

### Day Mode: Weighted Average

When both MOG2 and DIFF detect an object in day mode:

```python
gap = abs(mog2_time - diff_time)

if gap > 5.0:
    event_time = 0.6 * mog2_time + 0.4 * diff_time
else:
    event_time = mog2_time  # Close enough, trust MOG2
```

**Rationale**: MOG2 typically detects 1~5 seconds earlier than DIFF. When the gap is large (> 5s), averaging pulls the detection time closer to ground truth. When the gap is small, MOG2 alone is accurate enough.

The 3-second wait period ensures both methods have time to produce results before the system commits to a detection time.

## Person Path Heatmap

### Problem

MOG2 detects all foreground objects — not just abandoned items. Trees swaying, shadows moving, animals passing all generate blobs.

### Solution

Record where persons have walked and only consider blobs within those areas:

```python
# For each confirmed person track
for track in confirmed_persons:
    center = track.bbox_center
    cv2.circle(heatmap, center, radius=150, color=1, thickness=-1)

# When evaluating a blob
if heatmap[blob_center] == 0:
    skip  # No person has been here — ignore this blob
```

The 150px radius accounts for a person's reach when placing an object (arm length + placement motion).

### Person Confirmation

Not every YOLO detection is a real person. The hysteresis filter requires:

```
Person confirmed = 30 consecutive YOLO detections (~1 second at 30fps)
```

Additionally:
- Minimum total frames: 50
- Minimum dwell time: 5 seconds

This prevents:
- Nighttime 1-frame flickering from registering false person paths
- Brief misdetections (vehicles, animals) from contaminating the heatmap

## Leave Detection

### Two-Stage Process

**Stage 1: SUSPECT trigger (distance-based)**
```python
distance = euclidean(person_center, object_center)
if distance > 200:  # pixels
    state = SUSPECT
```

The person doesn't need to leave the frame — moving 200px away from the object is enough. This produces visually natural state transitions.

**Stage 2: Event timing (frame-exit based)**
```python
event_time = frame_exit_time + 10  # KISA 10-second offset
```

The actual event timestamp still uses the frame-exit time for timing accuracy, not the distance-based SUSPECT trigger time.

### Why Two Stages

Distance-based SUSPECT provides better visual feedback — the system shows "suspicious" as soon as the person walks away, not only after they leave the camera view. But the KISA timing criterion is based on object placement duration, so the event timestamp uses the more precise frame-exit time.

## Blob Lifecycle

```
DETECTED -> OBJECT (green) -> SUSPECT (orange) -> ABANDONED (red)
```

| State | Color | Condition |
|-------|-------|-----------|
| OBJECT | Green | Blob detected within person path |
| SUSPECT | Orange | Person moved > 200px from object |
| ABANDONED | Red | Event confirmed (placement + 10s) |

### Blob Filtering

| Filter | Condition | Purpose |
|--------|-----------|---------|
| Area minimum | MOG2 >= 150px, DIFF >= 100px | Ignore noise specks |
| Path check | Blob center within person heatmap | Ignore non-person-related objects |
| Age filter | Blob detected before person left - 30s | Ignore stale noise blobs |
| Size jump | Area change > 3x in one frame | Reject MOG2 noise merging |

### Bbox Refinement

After an object is confirmed (ABANDONED state), the system recalculates the bounding box using DIFF for better precision:

```python
# MOG2 often captures only part of the object
# DIFF reference-vs-current comparison gives cleaner boundaries
refined_bbox = compute_diff_bbox(reference_frame, current_frame, blob_region)
```

## Full Parameter Table

| Parameter | Value | Description |
|-----------|-------|-------------|
| conf | 0.25 | YOLO person confidence threshold |
| imgsz | 960 | YOLO input resolution |
| night_brightness_thr | 100 | Day/night brightness boundary |
| bg_warmup_sec | 30 | MOG2 background learning time |
| min_object_area_mog | 150 | MOG2 minimum blob area (px) |
| min_object_area_diff | 100 | DIFF minimum blob area (px) |
| diff_threshold | 20 | DIFF pixel intensity threshold |
| diff_confirm_frames | 3 | DIFF consecutive frame confirmation |
| path_radius | 150 | Person path heatmap radius (px) |
| person_gone_sec | 10 | Wait time after person leaves |
| person_away_dist | 200 | Distance-based leave threshold (px) |
| person_arrive_frames | 30 | Hysteresis (consecutive detections) |
| min_person_duration | 5s | Minimum person dwell time |
| min_person_frames | 50 | Minimum person detection frames |
| kisa_start_offset | 10 | KISA timing offset (+10s) |
| mog_static_confirm_sec | 10 | MOG2 blob confirmation time |
| person_margin_px | 20 | Person bbox masking margin |
| GaussianBlur | (5, 5) | DIFF blur kernel |
| weighted_avg_ratio | 0.6 / 0.4 | MOG2 / DIFF weight |
| day_wait | 3s | Wait for both results before commit |
| bbox_expansion_limit | 3x | MOG2 noise merge rejection |
| stale_blob_filter | 30s | Ignore blobs older than person exit - 30s |
