# Arson Detection — Troubleshooting

## Issue 1: Shared Memory Error on Training Server

**Error**: `RuntimeError: unable to open shared memory object` during data loading.

**Root Cause**: The training server has limited shared memory (`/dev/shm`). D-FINE's default `num_workers > 0` uses shared memory for inter-process data transfer.

**Solution**: Set `num_workers: 0` in the arson training config. This forces single-process data loading (slightly slower but avoids the memory issue).

**Note**: D-FINE configs use YAML `include` directives. The base `dataloader.yml` sets `num_workers: 4`, which overrides the arson config if not explicitly redefined. The fix is to set `num_workers: 0` directly in the arson-specific config file, not in the base config.

## Issue 2: CUDA Assertion Error in Matcher

**Error**: `CUDA error: device-side assert triggered` in `matcher.py` during the first training epoch.

**Root Cause**: The YOLO-format source dataset contained annotations with `class_id = 2`, which is outside the expected range for a 2-class model (0=person, 1=fire_smoke). When converted to COCO format, these annotations caused the Hungarian matcher to access invalid indices.

**Solution**: Added class ID filtering in `convert_yolo_to_coco.py`:
```python
cls_id = int(parts[0])
if cls_id not in valid_class_ids:
    continue  # Skip unexpected classes
```

**Lesson**: Always validate annotation class IDs against the model's class count before training.

## Issue 3: Config YAML Include Override

**Problem**: `num_workers` setting in the arson config was being silently overridden by `dataloader.yml`.

**Root Cause**: D-FINE uses a YAML include/merge system where base configs are loaded first, then task-specific configs are merged. The merge order meant `dataloader.yml` values took precedence.

**Solution**: Moved `num_workers: 0` to a position in the arson config that overrides the included base config.

## Issue 4: Domain Mismatch — 1st Training Failure

**Problem**: 1st-round model achieved mAP 92.3% on validation but completely failed on KISA CCTV footage. Fire confidence scores on KISA videos were indistinguishable from noise (0.03-0.08).

**Root Cause**: Training data was sourced from internet fire images (large, dramatic fires). KISA test footage shows small, distant fires from CCTV angles — completely different visual domain.

**Solution**: Collected KISA-domain data:
1. Extracted 1,510 frames from KISA arson videos (±30s around fire events)
2. Manually labeled fire/smoke in CVAT (732 fire + 181 smoke bbox)
3. Added Roboflow CCTV-specific fire/smoke datasets (13,000+ images)

**Lesson**: High validation mAP doesn't guarantee real-world performance. Domain-specific data is essential for CCTV applications.

## Issue 5: D-FINE Source Code Dependency

**Problem**: Initial attempt to use D-FINE via the DEXMA platform's module failed due to incomplete integration.

**Root Cause**: The DEXMA platform had a partial D-FINE integration missing required modules.

**Solution**: Cloned D-FINE directly from GitHub (`dfine/` directory, gitignored). Created standalone model wrapper (`dfine_loader.py`).

## Issue 6: D-FINE Output File Naming

**Problem**: Expected `best.pth`, but D-FINE outputs `best_stg2.pth`.

**Root Cause**: D-FINE uses two-stage training (Stage 1: detection, Stage 2: distillation/refinement). Final model is `best_stg2.pth`.

**Solution**: Documented convention. Deployment copies `best_stg2.pth` → `best.pth` locally.

## Issue 7: Nighttime Noise False Positives

**Problem**: Night CCTV footage produced massive false positives — fire noise bursts (15+ consecutive frames) passing the confirmation threshold, and >20 person detections per frame from noise.

**Root Cause**: Low-light CCTV produces high-noise frames that the model interprets as low-confidence fire/person detections. At conf=0.30, noise bursts lasting ~0.5s could pass the 5-frame confirmation threshold.

**Solution**: Dynamic day/night threshold system:
1. **Night detection**: brightness < 120 (immediate) or person noise > 50% of first 30 frames
2. **Night fire_confirm**: 5 → 30 frames (1-second burst can't pass)
3. **Night hard reset**: confirm_count = 0 on any miss frame (vs daytime soft decay -1)
4. **Night fire_conf**: 0.30 → 0.50 (filter low-confidence noise)
5. **Person noise skip**: >20 person detections per frame → skip person-in-zone check entirely

**Result**: C00_089 (nighttime video) changed from FAIL(-208s, noise false positive) to PASS(+1s).

## Issue 8: D-FINE Image Size Constraint

**Problem**: Attempted to change inference resolution from 640 to 960 for better small-fire detection.

**Error**: Positional embedding dimension mismatch.

**Root Cause**: D-FINE uses Transformer-based components with positional embeddings tied to the training resolution (640x640). Changing resolution at inference breaks the embedding dimensions.

**Solution**: Keep `imgsz=640` at both training and inference. To use a different resolution, the model must be retrained from scratch at that resolution.

## Issue 9: Camera Water Drops as Smoke

**Problem**: Rain/water drops on the camera lens were detected as smoke.

**Root Cause**: Water drops create blurry, semi-transparent patches similar to smoke's visual appearance. The model has no training data for this edge case.

**Status**: Known limitation. Would require water-drop negative examples in training data. Current impact is minimal since water drops rarely co-occur with person-in-zone detection (Rule #1 gate).