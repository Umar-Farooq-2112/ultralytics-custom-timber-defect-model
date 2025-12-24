## Training and Evaluation Results

This document summarizes the key quantitative results of training the **MobileNetV3-YOLO** model for timber defect detection.

### Source of Metrics

Metrics are taken from the Ultralytics training log CSV stored in this repository:

- `training-results/train/results.csv`

The log currently contains **114 epochs** of training (configured for up to 500 epochs with early stopping), with per-epoch losses and detection metrics.

### Overall Convergence

- **Training losses** (`train/box_loss`, `train/cls_loss`, `train/dfl_loss`) decrease steadily from very high values in early epochs to much lower, stable values by epoch 60–72.
- **Validation losses** (`val/box_loss`, `val/cls_loss`, `val/dfl_loss`) follow a similar downward trend, indicating that the model is learning meaningful representations without strong overfitting.

### Best Detection Metrics (Approximate)

From inspection of the `metrics/*` columns over the full 114 logged epochs:

- **Best mAP50 (bounding box)**: ~**0.79** (≈ 79%)
  - Achieved around **epoch 106**, where `metrics/mAP50(B)` peaks at about **0.79**.
- **Best mAP50-95 (bounding box)**: ~**0.46** (≈ 46%)
  - Also around epoch 106, where `metrics/mAP50-95(B)` is approximately **0.46**.
- **Precision**:
  - Increases from ~0.38 (epoch 1) to the **mid‑0.7 range** in later epochs.
- **Recall**:
  - Stabilizes around **0.77–0.78** in the best-performing region.

These numbers show that the custom MobileNetV3-YOLO architecture achieves **strong detection performance** on the timber defect dataset while remaining very lightweight.

### Qualitative Observations

Based on the metrics and model design:

- The enhanced P3 path and attention modules help the model **capture small surface defects** that would otherwise be missed.
- The lightweight backbone + neck combination provides a **good accuracy–efficiency trade-off**, making the model practical for real-time inspection scenarios.

### Checkpoints

The main checkpoints associated with this run are:

- `best.pt` – best-performing model (highest validation metric, typically mAP50).
- `last.pt` – weights from the last recorded training epoch.

These checkpoints live under the training run directory and can be loaded directly via Ultralytics:

```python
from ultralytics import YOLO

best_model = YOLO('path/to/best.pt')
last_model = YOLO('path/to/last.pt')
```

### Relation to Custom Code Repository

The architecture and training results documented here use the custom model code maintained at:

- https://github.com/Umar-Farooq-2112/ultralytics-custom-timber-defect-model.git

That repository contains the full MobileNetV3-YOLO implementation integrated into the Ultralytics framework.
