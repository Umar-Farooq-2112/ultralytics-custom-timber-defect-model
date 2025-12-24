## Custom Neck: UltraLiteNeckDW

The **UltraLiteNeckDW** module is a lightweight, attention-enhanced neck that converts backbone features into YOLO-ready multi-scale feature maps.

It is implemented in `ultralytics/nn/modules/custom_mobilenet_blocks.py` and is designed specifically for the **MobileNetV3BackboneDW** outputs.

### Inputs and Outputs

- **Inputs**: A list of three feature maps `[P3, P4, P5]` from `MobileNetV3BackboneDW`.
- **Input channels (typical)**: `[64, 128, 256]`.
- **Outputs**: Three fused feature maps for detection.
- **Output channels used by head**: `[128, 192, 256]` as specified in `MobileNetV3YOLO` (`neck_out_channels`).

### Core Components

1. **Depthwise-Separable Convolutions**
   - All main transformations use `DWConvCustom` to keep FLOPs low.
   - This maintains real-time performance even on modest GPUs/edge devices.

2. **CBAM_ChannelOnly Attention**
   - Applied per scale to refine features.
   - Channel attention (avg + max pooling) emphasizes salient filters.
   - Spatial attention focuses on relevant spatial regions (defect zones).

3. **SimSPPF (Simplified SPPF)**
   - Aggregates multi-scale context via repeated max pooling.
   - Increases receptive field without heavy computational cost.

4. **P5Transformer**
   - A tiny transformer encoder applied on the deepest feature map (P5).
   - Uses very small `embed_dim` and `ff_dim` to preserve efficiency.
   - Adds global context beneficial for elongated defects or context-dependent anomalies.

### Multi-Scale Fusion Strategy

While the exact implementation is in code, the conceptual flow is:

1. Process P3, P4, and P5 independently with DWConv + CBAM.
2. Apply SimSPPF and the P5Transformer on the deepest feature to capture global context.
3. Upsample and/or downsample features as needed to align spatial sizes.
4. Fuse (e.g., concat + conv) features across scales to form the final neck outputs sent to the `Detect` head.

This design mimics the behavior of PANet/FPN-style necks used in YOLO while being **much lighter**.

### Why This Neck is Good for Timber Defect Detection

- **Efficient yet expressive**: Depthwise convolutions and small transformers achieve good accuracy at low cost.
- **Attention-guided fusion**: CBAM ensures that important defect features are not diluted during fusion.
- **Context-aware P5**: The transformer on P5 improves recognition of complex defect patterns that depend on a larger context region.

For further details and source code, refer to the reference implementation:

- https://github.com/Umar-Farooq-2112/ultralytics-custom-timber-defect-model.git
