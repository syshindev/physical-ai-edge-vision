# Intrusion Detection — Legacy Code Analysis & Redesign

## Context

I inherited an existing intrusion detection codebase from a previous developer. While functional, it scored in the **low 80s** on the KISA national evaluation (150 videos). Through systematic analysis, I identified fundamental architectural limitations and rebuilt the detection pipeline, ultimately achieving **90+ points**.

## Legacy System Overview

| Aspect | Legacy | Redesigned |
|--------|--------|------------|
| **Lines of code** | 718 | 1,100+ |
| **Detection model** | torch manual load + manual NMS | Ultralytics YOLO (integrated pipeline) |
| **Tracker** | DeepSort (custom `nets.nn` module) | BoTSORT (built-in) + pseudo ID fallback |
| **State management** | Scattered dicts/sets | `TrackState` / `EventState` dataclasses |
| **ROI handling** | Single mode (area only) | Auto-switching area / strip modes |
| **Intrusion logic** | Simple "outside → inside" crossing | Streak-based + inside-ratio + confirmation delay |
| **Parameters** | 5 hardcoded | ~30 configurable |

## Problems Identified in Legacy Code

### 1. Fragile Tracking (DeepSort)

The legacy system used a custom DeepSort implementation (`nets.nn.DeepSort`). When the tracker lost a person's ID (common in CCTV with occlusions), the detection was simply lost — no recovery mechanism existed.

**Impact**: Missed intrusions when tracker ID switched mid-event.

**Fix**: Replaced with Ultralytics built-in BoTSORT and added a **pseudo ID matching** fallback. When BoTSORT returns `None` IDs, the system uses IoU-based matching against recent tracks to maintain continuity.

### 2. Binary Inside/Outside Logic

```python
# Legacy: simple 3-of-4-corners check
is_inside = self.is_bbox_mostly_in_polygon(box, zone_points)
if is_inside and track_id in self.previous_positions:
    # Intrusion detected!
```

This binary approach had no noise tolerance. A single frame where 3 corners fell inside the ROI would trigger detection, while slight bounding box jitter could cause missed detections.

**Impact**: Inconsistent detection — sensitive to bbox noise, especially at ROI boundaries.

**Fix**: Replaced with a multi-layered approach:
- **Streak counters**: Require N consecutive frames of "inside" status before confirming
- **Inside ratio**: Grid-sampling the bbox area to calculate what percentage is inside the ROI (threshold: 85%)
- **Confirmation delay**: Time-based delay (1.0s for area mode) to filter transient detections
- **Soft candidates**: Track partial entries (ratio ≥ 50%) as fallback candidates for video-end finalization

### 3. No ROI Shape Adaptation

The legacy system treated all ROI polygons identically. But KISA test videos include both:
- **Wide area ROIs** (e.g., parking lots, courtyards) — person walks INTO the zone
- **Thin strip ROIs** (e.g., fence lines, doorways) — person walks ACROSS the zone

Using area-mode logic on a thin strip ROI caused systematic failures: the "fully inside" check could never be satisfied because the person's body was always wider than the ROI.

**Impact**: Systematic false negatives on thin/strip ROI videos.

**Fix**: Built an **auto-detection system** that classifies ROI geometry at runtime:
```
ROI thickness < person_width × 2.5  →  Strip mode (crossing detection)
ROI thickness ≥ person_width × 2.5  →  Area mode (entry detection)
```
Each mode has dedicated logic, thresholds, and confirmation requirements.

### 4. No Event Lifecycle Management

```python
# Legacy: just record crossing time per track
self.intrusion_events_per_id[track_id] = {
    'crossing_time': crossing_time,
    'last_seen': current_elapsed_time
}
# At finalize: pick the last crossing
last = max(all_crossings, key=lambda x: x['crossing_time'])
```

Events were simply recorded as crossing timestamps. There was no concept of event duration, no hold period for temporary occlusions, and no mechanism to handle multiple intrusion events in a single video.

**Impact**:
- Brief occlusions split a single intrusion into multiple fragments
- No way to merge related events or select the correct one for KISA reporting

**Fix**: Introduced an `EventState` state machine:
```
Inactive → Event Start (person confirmed inside)
         → Event Hold (person temporarily lost, waiting end_hold_sec)
         → Event End (person definitively gone)
         → Finalize (select best event for KISA output)
```

Additional mechanisms:
- **Cooldown**: Prevent rapid re-triggering after event end
- **Hold period**: Keep event alive during brief tracking gaps
- **Event merging**: Merge events with < 10s gap (same intrusion, split by tracking loss)
- **KISA time correction**: Separate pre/post offsets for area and strip modes

### 5. Manual Preprocessing Overhead

```python
# Legacy: manual resize, tensor conversion, NMS
img, ratio, pad = self.resize(frame)
img_tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
outputs = self.model(img_tensor.unsqueeze(0))
outputs_nms = util.non_max_suppression(outputs, conf_thres, iou_thres)
# Manual coordinate scaling back to original size
```

The legacy code manually handled frame resizing, tensor conversion, NMS, and coordinate rescaling — ~50 lines of boilerplate that Ultralytics handles internally.

**Impact**: Maintenance burden, potential for subtle coordinate bugs.

**Fix**: Single unified call:
```python
results = self.model.track(
    source=frame, persist=True, tracker="botsort.yaml",
    classes=[0], conf=self.conf, iou=self.iou,
    imgsz=self.imgsz, verbose=False
)
```

### 6. Scattered State with No Cleanup

The legacy system used multiple independent data structures:
- `previous_positions` (dict) — last known positions
- `ignored_ids` (set) — IDs to skip
- `current_intruders` (set) — active intruders
- `intrusion_events_per_id` (dict) — recorded events
- `blacklisted_events` (dict) — fallback events

These had no TTL (time-to-live) or cleanup mechanism. Over a long video, stale entries accumulated indefinitely.

**Impact**: Memory growth and potential false matches with reused track IDs.

**Fix**: Consolidated into `TrackState` dataclass per track, with `_cleanup_stale_tracks()` removing entries that exceed `track_ttl_sec` (10s). Before cleanup, candidate states are preserved to prevent data loss.

## Results

| Metric | Legacy | Redesigned |
|--------|--------|------------|
| Dev test (30 videos) | ~24/30 PASS | **30/30 PASS** |
| KISA evaluation (150 videos) | Low 80s | **90+** |
| Timing accuracy | ±5–15s typical | ±1–6s typical |
| Thin ROI support | No | Yes (auto-detected) |
| Event merging | No | Yes (gap < 10s) |

## Key Takeaways

1. **Understand failure modes before coding**: Analyzing why the legacy system failed on specific videos guided every design decision
2. **State machines over ad-hoc flags**: The EventState lifecycle eliminated an entire class of edge-case bugs
3. **Adaptive over fixed thresholds**: Parameters that scale with input characteristics (person size, ROI shape) generalize better
4. **Fallback chains**: pseudo ID matching → soft candidates → force-confirm provides graceful degradation instead of hard failures
5. **Preserve backward compatibility**: The XML output format and `process_frame` / `create_result_xml` API remained identical, so the surrounding emulator code required zero changes