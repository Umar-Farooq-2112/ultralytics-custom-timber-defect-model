## Timber Defect Localization with MobileNetV3‑YOLO

This repository contains a **complete object-detection pipeline** for **localizing defects in timber boards** using a
custom, lightweight **MobileNetV3‑YOLO** model implemented on top of the Ultralytics YOLO framework.

The goal is that **anyone can understand and reproduce the project using this README alone**.

> Custom model reference implementation (original work):
> https://github.com/Umar-Farooq-2112/ultralytics-custom-timber-defect-model.git

---

## 1. Problem Statement and Goals

Wooden boards used in construction and furniture often contain **defects** such as cracks, knots, or surface damage.
These defects affect strength, safety, and appearance. Manually inspecting every board is:

- **Slow** – inspectors must visually scan large surfaces.
- **Subjective** – defect severity can be judged differently by different people.
- **Expensive** – especially for high-volume production lines.

This project builds an **automated vision system** that:

- Takes an **RGB image of a timber board** as input.
- Outputs **bounding boxes around visible defects** with class labels and confidence scores.
- Is **lightweight enough** to run on limited hardware (e.g., edge devices or low-cost GPUs).

The end result is a deployable model that can be integrated into a factory pipeline for **automatic timber defect
localization**.

---

## 2. High‑Level Overview

At a high level, the project consists of:

1. **Dataset & Annotations**
   - A defect dataset of timber boards (custom dataset, 4 defect classes).
   - Labels stored in standard YOLO format via a `data.yaml` file.

2. **Custom Detection Model – MobileNetV3‑YOLO**
   - **Backbone**: `MobileNetV3BackboneDW` (based on MobileNetV3‑Small) with depthwise separable convolutions and
     attention.
   - **Neck**: `UltraLiteNeckDW` – an extremely lightweight feature pyramid network with attention, SimSPPF, and a tiny
     transformer block.
   - **Head**: Standard Ultralytics YOLO `Detect` head for bounding box regression and classification.

3. **Training & Evaluation**
   - Training is driven by a custom model YAML:
     - `ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml` (with `nc: 4`).
   - Training configuration is stored in:
     - `training-results/train/args.yaml`.
   - Training metrics (loss, mAP, precision/recall) are stored in:
     - `training-results/train/results.csv` and `training-results/train/processed_results.csv`.

4. **Documentation & Notebooks**
   - `docs/architecture.md` – full model architecture description.
   - `docs/custom_backbone.md` – details of the MobileNetV3 backbone.
   - `docs/custom_neck.md` – details of the UltraLite neck.
   - `docs/training_pipeline.md` – training pipeline settings and rationale.
   - `docs/results.md` – summarized results and interpretation.
   - `notebooks/` – placeholders for data augmentation and training visualization.

---

## 3. Repository Structure (Project‑Relevant Parts)

Only the parts directly related to the timber defect project are listed here:

```text
ultralytics-custom-backbone-neck/
├─ ultralytics/
│  ├─ cfg/models/custom/mobilenetv3-yolo.yaml   # Custom model configuration (4 defect classes)
│  ├─ nn/custom_models.py                       # MobileNetV3YOLO model class
│  └─ nn/modules/custom_mobilenet_blocks.py     # Backbone and neck building blocks
│
├─ docs/
│  ├─ architecture.md                           # End‑to‑end architecture of MobileNetV3‑YOLO
│  ├─ custom_backbone.md                        # MobileNetV3BackboneDW design
│  ├─ custom_neck.md                            # UltraLiteNeckDW design
│  ├─ training_pipeline.md                      # Training configuration and procedure
│  └─ results.md                                # Quantitative and qualitative results
│
├─ training-results/
│  └─ train/
│     ├─ args.yaml                              # Exact hyperparameters used in the final run
│     ├─ results.csv                            # Raw training metrics per epoch
│     └─ processed_results.csv                  # Cleaned metrics for analysis/plots
│
├─ notebooks/
│  ├─ data_augnmentation.ipynb                  # To document data augmentation pipeline
│  ├─ model_training_and_results.ipynb          # To visualize training curves and metrics
│  └─ sample/                                   # (Optional) Tiny sample data or demos
│
├─ best.pt                                      # Best performing checkpoint (validation‑based)
├─ last.pt                                      # Last checkpoint from the recorded training run
└─ README.md                                    # You are here
```

Other files and folders belong to the upstream Ultralytics framework (data loaders, generic models, training engine
etc.). They are used as a base but are not the main focus of this project description.

---

## 4. Dataset and Labels

### 4.1. Dataset

The project uses a **custom timber defect dataset** described via a standard Ultralytics `data.yaml` file. In the
recorded experiments, the dataset is referenced as:

```text
defects-in-timber/data.yaml
```

This file defines:

- Paths to **training**, **validation**, and (optionally) **test** image folders.
- A list of **4 defect classes** (e.g., cracks, knots, surface anomalies – exact names depend on your dataset).
- Any dataset‑specific metadata.

The raw images are photos of timber boards taken under controlled lighting. Each defect is annotated as a **bounding
box** with a class label.

### 4.2. Annotation Format

Ultralytics (YOLO) uses a standard format:

- Each image has a corresponding `.txt` file with one line per object.
- Each line contains: `class_id x_center y_center width height` in **normalized** coordinates (values 0–1).
- Class indices range from `0` to `3` for the four defect types.

The `data.yaml` file tells the training code **where** these images and labels are stored.

---

## 5. Model Architecture – MobileNetV3‑YOLO

The model follows a standard YOLO design (backbone → neck → detection head) but is heavily optimized for **lightweight
inference** and **small defect localization**.

### 5.1. Backbone – MobileNetV3BackboneDW

Defined in: `ultralytics/nn/modules/custom_mobilenet_blocks.py` and wired in via `MobileNetV3YOLO`.

Main ideas:

- **Base network**: `mobilenet_v3_small` from `torchvision`.
- The network is sliced into stages to produce 3 main feature maps:
  - `P3` at stride 8 (high resolution, good for small defects).
  - `P4` at stride 16.
  - `P5` at stride 32.
- Custom **depthwise separable convolution blocks** (`DWConvCustom`) are used to reduce parameters and FLOPs.
- A **CBAM‑like attention module** (channel attention) is added to improve feature quality for defect regions.

Why this backbone?

- MobileNetV3‑Small is already optimized for mobile/edge hardware.
- Adding depthwise blocks and attention improves performance on difficult, small, low‑contrast defects while keeping
  the model compact.

### 5.2. Neck – UltraLiteNeckDW

Defined in: `ultralytics/nn/modules/custom_mobilenet_blocks.py`.

The neck combines features from P3, P4, and P5 and prepares them for detection:

- Uses **depthwise separable convolutions** throughout to keep computation low.
- Incorporates **SimSPPF** (a simplified spatial pyramid pooling) to capture multi‑scale context.
- Adds a small **transformer block on the P5 layer** to model long‑range dependencies across the board.
- Applies **channel attention** to emphasize informative channels.

Outputs of the neck are 3 feature maps (for small, medium, and large objects) that are fed into the YOLO `Detect` head.

### 5.3. Detection Head – YOLO `Detect`

The head is the standard Ultralytics YOLO detection head:

- For each scale (P3, P4, P5), it predicts:
  - Bounding box coordinates (using a Distribution Focal Loss‑based representation).
  - Objectness score.
  - Class probabilities for the 4 defect classes.
- Loss is computed via Ultralytics' detection loss, handling localization and classification jointly.

### 5.4. Capacity and Efficiency

- Total parameter count is **on the order of ~1.5M parameters** (much smaller than YOLOv8n).
- Designed to run on modest GPUs and potentially edge devices.
- Suitable for **real‑time or near real‑time** inspection depending on hardware.

For diagrams and more detailed layer descriptions, see:

- `docs/architecture.md`
- `docs/custom_backbone.md`
- `docs/custom_neck.md`

---

## 6. Training Configuration

All training hyperparameters for the main experiment are stored in:

- `training-results/train/args.yaml`

Key settings from this file:

- **Task**: `detect`
- **Model config**: `ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml`
- **Data config**: `defects-in-timber/data.yaml`
- **Classes**: 4
- **Image size**: `imgsz = 640`
- **Batch size**: `batch = 16`
- **Epochs**: configured for `epochs = 500`
- **Early stopping**: `patience = 25` (training stops early if validation does not improve)
- **Optimizer**: SGD with typical YOLOv8 defaults (initial learning rate, momentum, weight decay, etc.).
- **Augmentations**: standard Ultralytics augmentations (random scaling, flipping, color jitter, mosaic/mixup depending
  on the exact version).

These parameters make the training **stable** while allowing the model enough capacity to learn subtle texture
differences on wood.

For a narrative explanation of the pipeline, see `docs/training_pipeline.md`.

---

## 7. Training Results

The detailed per‑epoch metrics are logged in:

- `training-results/train/results.csv` (raw Ultralytics log)
- `training-results/train/processed_results.csv` (cleaned version for analysis)

The recorded training run logs **114 epochs** (with configuration set to a maximum of 500 epochs and early stopping).

From these logs (see `docs/results.md` for plots and interpretation):

- **Best mAP50 (bounding box)**: ≈ **0.79** (around epoch ~106)
- **Best mAP50‑95 (bounding box)**: ≈ **0.46**
- **Precision**: grows from ~0.38 in early epochs to the **mid‑0.7 range** for the best checkpoint
- **Recall**: stabilizes around **0.77–0.78**

Checkpoint files:

- `best.pt` – model with best validation performance.
- `last.pt` – model from the final logged epoch (useful for continued training or analysis).

These results show that the custom MobileNetV3‑YOLO successfully learns to detect multiple defect types with solid
precision and recall, while remaining much smaller than standard YOLO variants.

---

## 8. How to Run This Project

This section explains **exactly how to install, train, and run inference** using this repository.

### 8.1. Requirements

- Python >= 3.8
- PyTorch with CUDA support (if you want GPU training)
- Git

### 8.2. Installation (Editable Mode)

1. Clone this repository:

   ```bash
   git clone <this-repo-url>
   cd ultralytics-custom-backbone-neck
   ```

2. Create and activate a virtual environment (recommended) and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows

   pip install --upgrade pip
   pip install -e .
   ```

   This installs the `ultralytics` package from this repo, including the custom MobileNetV3‑YOLO model.

### 8.3. Prepare the Dataset

1. Place your timber defect images and labels in a folder structure compatible with Ultralytics YOLO.
2. Create or adapt a `data.yaml` similar to the one used in experiments:

   ```yaml
   path: path/to/defects-in-timber
   train: images/train
   val: images/val
   test: images/test  # optional

   names:
     0: class_0
     1: class_1
     2: class_2
     3: class_3
   ```

   Replace `class_0`–`class_3` with your actual defect names.

Update `training-results/train/args.yaml` or simply pass your data path on the command line.

### 8.4. Training from Scratch

Once dependencies and dataset are ready, train the model using the `yolo` CLI:

```bash
yolo detect train \
  model=ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml \
  data=defects-in-timber/data.yaml \
  imgsz=640 \
  batch=16 \
  epochs=500 \
  patience=25
```

During training, Ultralytics will automatically create a new run directory (e.g., `runs/detect/trainX/`) with logs,
plots, and checkpoints. The run used in this README is also summarized under `training-results/train/`.

### 8.5. Evaluating a Trained Model

To evaluate an existing checkpoint (for example, `best.pt`):

```bash
yolo detect val \
  model=best.pt \
  data=defects-in-timber/data.yaml \
  imgsz=640
```

This reports mAP, precision, recall, and other metrics on the validation set.

### 8.6. Running Inference on New Images

To run inference on one or more images of timber boards:

```bash
yolo detect predict \
  model=best.pt \
  source=path/to/your/images \
  imgsz=640 \
  conf=0.25
```

Ultralytics will create a `runs/detect/predictX/` folder containing images with **bounding boxes and labels drawn on
detected defects**.

### 8.7. Using the Model in Python

You can also integrate the trained model directly in Python code:

```python
from ultralytics import YOLO

# Load a trained timber defect model
model = YOLO("best.pt")

# Run prediction on a single image
results = model("path/to/board.jpg")

# Visualize
results[0].show()             # show in a window
results[0].save("out.jpg")   # save visualization
```

---

## 9. Notebooks and Analysis

Two notebooks are provided as **templates** to turn this project into a fully documented experiment:

- `notebooks/data_augnmentation.ipynb`
  - Document and visualize the data augmentation pipeline.
  - Show example before/after images.

- `notebooks/model_training_and_results.ipynb`
  - Load `training-results/train/processed_results.csv`.
  - Plot loss, mAP, precision, and recall curves.
  - Compare different runs if you perform further experiments.

You can adapt these notebooks to your own environment and dataset paths.

---

## 10. Why This Approach Works Well for Timber Defects

1. **Small, localized defects**
   - Many defects are small relative to the entire board.
   - The strong P3 (stride 8) path and attention modules help retain high‑resolution detail.

2. **Lightweight but expressive backbone**
   - MobileNetV3‑Small is compact yet expressive.
   - Depthwise convolutions make it feasible for real‑time use.

3. **Context through the neck**
   - SimSPPF and the tiny transformer on P5 give the model enough global context to distinguish real defects from
     harmless texture.

4. **Stable training pipeline**
   - Standard Ultralytics training loop, optimizer, and augmentations.
   - Easy to reproduce and extend with new experiments.

---

## 11. Extending the Project

Ideas for future work:

- **Per‑defect severity scoring**: add regression heads or additional outputs to estimate severity or size.
- **Segmentation**: extend the model to instance or semantic segmentation of defect regions.
- **Active learning**: use model uncertainty to propose new images for labeling.
- **Edge deployment**: export to ONNX, TensorRT, or other formats and deploy on embedded devices.

The existing codebase and configuration files already follow the Ultralytics ecosystem, so most extensions can reuse
their tooling.

---

## 12. License and Acknowledgements

- The core training framework and many utilities come from the **Ultralytics YOLO** project.
- This repository adds a custom backbone, neck, and configuration for the **timber defect localization** use case.

Please see the [LICENSE](LICENSE) file for full licensing details.

If you use this work in academic research, consider citing both the Ultralytics YOLO paper and your own thesis/paper
that describes the timber defect model.

---

## 13. Contact

For questions about **this timber defect project** (model design, training configuration, or results), please refer to
the documentation in the `docs/` folder or to the original custom model repository:

- https://github.com/Umar-Farooq-2112/ultralytics-custom-timber-defect-model.git

You can also consult the official Ultralytics documentation for general YOLO usage patterns and advanced deployment
options.

