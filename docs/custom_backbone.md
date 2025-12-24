## Custom Backbone: MobileNetV3BackboneDW

The **MobileNetV3BackboneDW** class (in `ultralytics/nn/modules/custom_mobilenet_blocks.py`) adapts the standard `mobilenet_v3_small` network from Torchvision into a multi-scale detection backbone optimized for timber defect detection.

### Design Goals

- Maintain **very low parameter count** and FLOPs.
- Strengthen **shallow and mid-level features** to capture small timber defects.
- Provide clean, YOLO-style outputs at strides 8, 16, and 32.

### Stage Layout

Starting from `mobilenet_v3_small(pretrained=True)`, the backbone is sliced into three stages:

- `stage1 = feats[:3]` → **P3/8** (24 channels)
- `stage2 = feats[3:7]` → **P4/16** (40 channels)
- `stage3 = feats[7:]` → **P5/32** (576 channels)

These stages correspond to progressively downsampled feature maps used by the neck.

### Depthwise Separable Blocks (DWConvCustom)

The backbone enhances each stage with `DWConvCustom`, a depthwise-separable convolution block:

- Depthwise 3×3 convolution (per-channel spatial filtering).
- Pointwise 1×1 convolution (channel mixing).
- Optional BatchNorm and SiLU activation.

For P3 in particular, the backbone applies a **stack of multiple DWConvCustom blocks**:

- `conv_p3_1`: 24 → 48 channels
- `conv_p3_2`: 48 → 64 channels
- `conv_p3_3`: 64 → 64 channels
- `conv_p3_4`: 64 → 64 channels
- `conv_p3_5`: 64 → 64 channels

This deeper P3 path gives the model enough capacity to describe **fine-grained surface defects** without significantly increasing the parameter count.

### CBAM_ChannelOnly Attention

The backbone uses `CBAM_ChannelOnly` to refine features:

- **Channel attention** via both average and max pooling.
- **Spatial attention** over concatenated average and max feature maps.
- Multiple 1×1 convolutions with SiLU to learn non-linear attention maps.

On P3, CBAM helps highlight small crack or knot regions; on deeper stages it enhances discriminative defect patterns.

### Output Channels

After backbone enhancement and attention, the backbone provides feature maps with channels approximately:

- P3: 64 channels
- P4: 128 channels (after additional DWConvCustom blocks on the 40-channel MobileNetV3 feature)
- P5: 256 channels (after reducing the 576-channel feature to a compact representation)

These channels are then passed to `UltraLiteNeckDW` for further fusion and adaptation to the YOLO head.

### Why This Backbone is Good for Timber Defects

- **Pretrained initialization**: MobileNetV3 Small weights transfer general visual knowledge to the timber domain, accelerating convergence.
- **Depthwise separable design**: Efficient enough for real-time inspection systems.
- **Enhanced P3 path**: Focuses capacity on high-resolution features where small defects appear.
- **CBAM attention**: Encourages the network to attend to defect regions rather than background wood texture.

For full implementation details, see the reference repository:

- https://github.com/Umar-Farooq-2112/ultralytics-custom-timber-defect-model.git
