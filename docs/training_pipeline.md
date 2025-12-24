## Training Pipeline for Timber Defect Detection

This document summarizes how the **MobileNetV3-YOLO** model is integrated into the Ultralytics training pipeline for timber defect localization.

### 1. Environment and Dependencies

- Base project: this repository (Ultralytics fork with custom backbone/neck).
- Python: ≥ 3.8.
- Deep learning framework: PyTorch (as required by Ultralytics).
- Package management: `pyproject.toml` at the root defines the main dependencies.

Install Ultralytics and dependencies (from this repo) in a suitable environment before training.

### 2. Custom Model Configuration

- Custom YAML: `ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml`.
- Key fields:
  - `nc: 4` – number of timber defect classes.
  - `custom_model: mobilenetv3-yolo` – identifier used by `parse_custom_model` to build the model.
  - `scales` – placeholder values; scaling is not used because the backbone channels are fixed.

Model construction is handled programmatically in:

- `ultralytics/nn/custom_models.py` → `MobileNetV3YOLO` class.

### 3. Training Entry Points

Typical training is performed using:

- `train_custom_model.py` (custom training script in this repo).
- Or the standard Ultralytics Python/CLI APIs:
  - CLI: `yolo detect train model=ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml data=path/to/data.yaml`.
  - Python:
    - `from ultralytics import YOLO`
    - `model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')`
    - `model.train(data='path/to/data.yaml', epochs=..., imgsz=640, ...)`.

The **DetectionTrainer** and loss functions are reused from Ultralytics, ensuring full compatibility with existing workflows.

### 4. Dataset and Augmentation (High-Level)

The dataset and augmentations are prepared in the separate project folder:

- `Hybrid-YOLO-Vision-transformers-and-Graph-Neural-Network-based-Timber-defect-localization/`

Augmentation steps (performed in that project) typically include:

- Geometric transforms (flip, rotate, scale).
- Color jitter and illumination variations.
- Cropping and multi-scale resizing.
- Optional advanced augmentations like mosaic or mixup for robustness.

The resulting dataset is used here via a standard Ultralytics YAML (`data.yaml`), which defines:

- Paths to training/validation images.
- Class names and number of classes (4).

### 5. Logging and Metrics

During training, Ultralytics automatically logs metrics per epoch to CSV files. For this project, the key files are stored directly in this repository under:

- `training-results/train/results.csv` – raw Ultralytics training log.
- `training-results/train/processed_results.csv` – same log with friendlier column names.
- `training-results/train/args.yaml` – full training configuration (model, data YAML, epochs, batch size, augmentation, etc.).

In the recorded training run, `args.yaml` shows:

- `model: ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml`
- `data: defects-in-timber/data.yaml` (dataset configuration)
- `epochs: 500` with `patience: 25` for early stopping
- `batch: 16`, `imgsz: 640`, and standard Ultralytics augmentation settings

Important columns from `results.csv`/`processed_results.csv` include:

- `train/box_loss`, `train/cls_loss`, `train/dfl_loss` – training losses.
- `val/box_loss`, `val/cls_loss`, `val/dfl_loss` – validation losses.
- `metrics/precision(B)`, `metrics/recall(B)` – detection precision and recall.
- `metrics/mAP50(B)`, `metrics/mAP50-95(B)` – main detection metrics.

### 6. Validation and Testing

Validation is performed automatically at the end of each epoch by Ultralytics. The final model is typically selected based on **best mAP50 or mAP50-95**.

Integration tests for the custom model (sanity checks) are provided in this repository:

- `test_integration.py` – checks imports, YAML loading, forward pass, info, `parse_custom_model`, and a dry-run training integration.

### 7. Checkpointing

The training run saves:

- `best.pt` – best-performing weights according to the chosen metric (usually mAP50 or mAP50-95).
- `last.pt` – weights from the last training epoch.

These checkpoints can be used for:

- Further fine-tuning.
- Inference in production (after optional layer fusion).
- Export to ONNX, TensorRT, etc., via the standard Ultralytics export APIs.

### 8. Inference Workflow (High-Level)

After training, inference with the custom model follows the standard YOLO workflow:

```python
from ultralytics import YOLO

model = YOLO('best.pt')  # trained MobileNetV3-YOLO checkpoint
results = model('path/to/timber_image_or_folder')
```

The model outputs bounding boxes and class predictions for timber defects. Post-processing and visualization follow standard Ultralytics APIs.
