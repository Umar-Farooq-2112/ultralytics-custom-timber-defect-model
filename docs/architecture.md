## Timber Defect Detection Architecture (MobileNetV3-YOLO)

This project integrates a lightweight custom object detector for timber defect localization into the Ultralytics YOLO codebase. The final model is a **MobileNetV3-YOLO** variant designed for small defect detection on lumber surfaces with a strong balance between accuracy, speed, and parameter efficiency.

### High-Level Design

- **Task**: Single-stage object detection for timber defects (bounding boxes).
- **Base framework**: Ultralytics YOLO (v8-style detection pipeline and loss).
- **Custom model**: `MobileNetV3YOLO` defined in `ultralytics/nn/custom_models.py`.
- **Configuration entry point**: `ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml`.
- **Number of classes (nc)**: 4 (timber defect classes as defined in the project dataset).

The architecture is split into three main components:

1. **Backbone** – `MobileNetV3BackboneDW`
2. **Neck** – `UltraLiteNeckDW`
3. **Head** – standard YOLOv8-style `Detect` head

All components are implemented as PyTorch modules and are fully compatible with the standard Ultralytics training, validation, export, and inference APIs.

### Backbone: MobileNetV3 Small with Depthwise Enhancements

The backbone starts from the torchvision **MobileNetV3 Small** classifier and converts it into a detection-friendly feature extractor:

- Uses `mobilenet_v3_small(pretrained=True)` from `torchvision.models`.
- Feature stages are sliced into three outputs:
  - **P3 (stride 8)**: `feats[:3]` → 24 channels.
  - **P4 (stride 16)**: `feats[3:7]` → 40 channels.
  - **P5 (stride 32)**: `feats[7:]` → 576 channels.
- Each stage is further processed using depthwise-separable blocks (`DWConvCustom`) and attention (`CBAM_ChannelOnly`) to better preserve fine-grained defect features.

This design preserves MobileNetV3's **efficiency** while strengthening the **low-level representation** that is critical for small cracks, knots, and surface anomalies on timber.

### Neck: UltraLiteNeckDW

The neck (`UltraLiteNeckDW`) performs **multi-scale fusion** of the backbone features and prepares them for the YOLO detection head. Key elements include:

- Depthwise separable convolutions for all main paths to reduce FLOPs and parameters.
- **CBAM_ChannelOnly** modules for channel + spatial attention at each scale.
- **SimSPPF** (simplified SPPF) on the highest-resolution feature to aggregate multi-scale context.
- A compact **P5Transformer** on the deepest feature map to improve global context while keeping the embedding dimension very small.

The neck outputs three feature maps used by the head:

- P3: 128 channels (after fusion)
- P4: 192 channels
- P5: 256 channels

These are configured via `neck_out_channels = [128, 192, 256]` inside `MobileNetV3YOLO`.

### Head: YOLOv8-Style Detect

The project reuses the standard Ultralytics `Detect` head:

- Input channels: `(128, 192, 256)` corresponding to `(P3, P4, P5)`.
- Outputs per anchor: bounding box regression, objectness, and class logits.
- Uses **Distribution Focal Loss (DFL)** for bounding box quality.

Because the head is unchanged, the model benefits from **mature, well-tested YOLO loss functions and NMS pipelines** while customizing only the backbone and neck.

### Strides and Feature Map Sizes

For an input image of `640 × 640`:

- P3: stride 8 → `80 × 80` feature map.
- P4: stride 16 → `40 × 40` feature map.
- P5: stride 32 → `20 × 20` feature map.

These three feature levels are used together to detect both **small local defects** (on P3/P4) and **larger structural issues** (on P5).

### Model Size and Efficiency

According to the comments in `mobilenetv3-yolo.yaml` and the design of the custom modules:

- Total parameters: **~1.5M** (approximately half of YOLOv8n).
- FLOPs: **significantly lower** than a standard YOLOv8n detector at the same input size.

This makes the model well-suited for **edge devices**, **industrial cameras**, and **real-time inspection pipelines**.

### Custom Model Reference

The custom backbone and neck code used here is maintained in the following reference repository:

- **GitHub**: https://github.com/Umar-Farooq-2112/ultralytics-custom-timber-defect-model.git

This repository provides the full source for the custom MobileNetV3-YOLO architecture integrated into Ultralytics.
