# Arson Detection — Model Migration (RT-DETR → D-FINE Nano)

## Why Migrate

| Factor | RT-DETR | D-FINE Nano |
|--------|---------|-------------|
| **Architecture** | Transformer-based | HGNetv2 B0 backbone |
| **Platform** | Ultralytics (standalone) | Shared with DEXMA platform |
| **Tracking** | Required BoTSORT | Not needed (no tracking required for arson) |
| **Model size** | Larger | Smaller (Nano variant) |
| **Reason** | Legacy implementation | Platform unification with DEXMA |

The migration was driven by the need to unify the detection backend with the DEXMA hybrid AI platform, which already used D-FINE for other detection tasks. Maintaining two different model architectures (RT-DETR for arson, D-FINE for others) created unnecessary complexity.

## Migration Steps

### Step 1: Model Wrapper

Created `DFineModel` class (`dfine_loader.py`) as a drop-in replacement for the Ultralytics model interface:

```python
class DFineModel:
    def __init__(self, config, weights, device, conf, imgsz):
        # Load D-FINE model from config + weights
        ...

    def predict(self, frame) -> list:
        # Returns [[x1, y1, x2, y2, conf, class_id], ...]
        ...
```

The wrapper standardizes the output format so the detection module doesn't need to know which model is running underneath.

### Step 2: Import Changes

```python
# Before (RT-DETR)
from ultralytics import RTDETR
model = RTDETR(weights)
results = model.track(frame, ...)

# After (D-FINE)
from dfine_loader import DFineModel
model = DFineModel(config=config, weights=weights, ...)
detections = model.predict(frame)
```

### Step 3: Tracker Removal

RT-DETR used BoTSORT tracking to maintain person IDs across frames. For arson detection, tracking is unnecessary — we only need to know if a person/fire exists in the current frame, not maintain identity across frames. Removing BoTSORT:

- Eliminated the `model.track()` call (replaced with `model.predict()`)
- Removed BoTSORT predictor reset code
- Simplified the detection pipeline

### Step 4: Dual-Threshold Inference

D-FINE outputs both person and fire detections in a single forward pass. The challenge: fire/smoke confidence scores are typically much lower than person scores (the model is less confident about fire).

**Solution**: Two separate confidence thresholds with dynamic day/night adjustment:
- `conf = 0.05` for person detection (small/distant persons in CCTV)
- `fire_conf = 0.30` for fire/smoke detection (daytime; 0.50 at night)
- Model runs at `min(conf, fire_conf)` threshold internally
- Class-specific filtering happens post-inference

### Step 5: Config Parameter Addition

The D-FINE model requires a YAML config file (defining architecture, class count, etc.) in addition to weights:

```python
# ArsonMonitor.__init__ now takes a config parameter
def __init__(self,
    config: str = "configs/dfine/custom/dfine_hgnetv2_n_arson.yml",
    weights: str = "weights/arson_dfine/best.pth",
    ...):
```

## Verification

After migration:
1. Ran all test videos to confirm detection behavior
2. Verified fire/smoke detection sensitivity with the low `fire_conf` threshold
3. Confirmed person-in-zone detection works with the new model
4. Checked GPU memory usage (D-FINE Nano uses less memory)

## Code Changes Summary

| File | Change |
|------|--------|
| `arson.py` | RT-DETR → DFineModel import, removed BoTSORT, added config param, dual thresholds |
| `dfine_loader.py` | New file — D-FINE model wrapper |
| `configs/dfine/custom/` | New D-FINE model config for arson |