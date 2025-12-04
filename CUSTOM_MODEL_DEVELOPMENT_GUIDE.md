# Custom Model Development Guide

## How to Safely Modify Your Backbone, Neck, or Head

This guide shows you **step-by-step** how to modify your custom model architecture safely without breaking the training pipeline.

---

## Table of Contents

1. [Understanding the Architecture](#understanding-the-architecture)
2. [Modifying the Backbone](#modifying-the-backbone)
3. [Modifying the Neck](#modifying-the-neck)
4. [Modifying the Head](#modifying-the-head)
5. [Testing Your Changes](#testing-your-changes)
6. [Common Modification Patterns](#common-modification-patterns)
7. [Troubleshooting](#troubleshooting)

---

## Understanding the Architecture

### Current Model Structure

```
Input (B, 3, 640, 640)
    ↓
┌─────────────────────┐
│   MobileNetV3       │  ← BACKBONE
│   Backbone          │     Extracts features
└─────────────────────┘
    ↓
  [P3, P4, P5]           Channels: [24, 40, 160]
    ↓
┌─────────────────────┐
│   UltraLite         │  ← NECK
│   Neck              │     Fuses features
└─────────────────────┘
    ↓
  [P3, P4, P5]           Channels: [32, 48, 64]
    ↓
┌─────────────────────┐
│   Detect            │  ← HEAD
│   Head (YOLOv8)     │     Generates predictions
└─────────────────────┘
    ↓
Output: 3 detection layers
```

### Key Files

- **`ultralytics/nn/modules/custom_mobilenet_blocks.py`** - All custom modules (backbone, neck, blocks)
- **`ultralytics/nn/custom_models.py`** - Main model class that connects everything
- **`ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml`** - Model configuration

---

## Modifying the Backbone

### Example: Add More Layers to MobileNetV3 Backbone

**Location:** `ultralytics/nn/modules/custom_mobilenet_blocks.py`

#### Step 1: Understand Current Output Channels

Current backbone outputs:
- **P3**: 24 channels (stride 8)
- **P4**: 40 channels (stride 16)
- **P5**: 160 channels (stride 32)

#### Step 2: Modify the Backbone Class

**Find the `MobileNetV3BackboneDW` class:**

```python
class MobileNetV3BackboneDW(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.features = mobilenet.features
        
        # Define depthwise convolutions for each stage
        self.conv_p3_dw = DWConvCustom(24, 24, k=3, s=1)
        self.conv_p4_dw = DWConvCustom(40, 40, k=3, s=1)
        self.conv_p5_dw = DWConvCustom(160, 160, k=3, s=1)
        
        # Stage indices in MobileNetV3 Small
        self.stage1 = nn.Sequential(*self.features[:2])   # → P3 (24 channels)
        self.stage2 = nn.Sequential(*self.features[2:4])  # → P4 (40 channels)
        self.stage3 = nn.Sequential(*self.features[4:])   # → P5 (160 channels)
```

#### Step 3: Add Extra Layers

**Option A: Add extra depthwise convolutions**

```python
class MobileNetV3BackboneDW(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.features = mobilenet.features
        
        # Original convolutions
        self.conv_p3_dw = DWConvCustom(24, 24, k=3, s=1)
        self.conv_p4_dw = DWConvCustom(40, 40, k=3, s=1)
        self.conv_p5_dw = DWConvCustom(160, 160, k=3, s=1)
        
        # ADD: Extra refinement layers
        self.extra_p3 = DWConvCustom(24, 24, k=3, s=1)    # ← NEW
        self.extra_p4 = DWConvCustom(40, 40, k=3, s=1)    # ← NEW
        self.extra_p5 = DWConvCustom(160, 160, k=3, s=1)  # ← NEW
        
        self.stage1 = nn.Sequential(*self.features[:2])
        self.stage2 = nn.Sequential(*self.features[2:4])
        self.stage3 = nn.Sequential(*self.features[4:])
    
    def forward(self, x):
        # Extract features
        p3 = self.stage1(x)
        p4_in = self.stage2(p3)
        p4 = self.conv_p4_dw(p4_in)
        p5_in = self.stage3(p4)
        p5 = self.conv_p5_dw(p5_in)
        
        # ADD: Apply extra layers
        p3 = self.extra_p3(p3)    # ← NEW
        p4 = self.extra_p4(p4)    # ← NEW
        p5 = self.extra_p5(p5)    # ← NEW
        
        return [p3, p4, p5]
```

**Option B: Change output channels**

```python
class MobileNetV3BackboneDW(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.features = mobilenet.features
        
        # CHANGE: Increase output channels
        self.conv_p3_dw = DWConvCustom(24, 32, k=3, s=1)    # 24 → 32 ← CHANGED
        self.conv_p4_dw = DWConvCustom(40, 64, k=3, s=1)    # 40 → 64 ← CHANGED
        self.conv_p5_dw = DWConvCustom(160, 128, k=3, s=1)  # 160 → 128 ← CHANGED
        
        self.stage1 = nn.Sequential(*self.features[:2])
        self.stage2 = nn.Sequential(*self.features[2:4])
        self.stage3 = nn.Sequential(*self.features[4:])
    
    def forward(self, x):
        p3 = self.stage1(x)
        p4_in = self.stage2(p3)
        p4 = self.conv_p4_dw(p4_in)
        p5_in = self.stage3(p4)
        p5 = self.conv_p5_dw(p5_in)
        
        return [p3, p4, p5]  # Now: [32, 64, 128] instead of [24, 40, 160]
```

#### Step 4: Update the Neck Input Channels

**⚠️ CRITICAL:** If you changed backbone output channels, update the neck!

**Location:** `ultralytics/nn/custom_models.py`

```python
class MobileNetV3YOLO(nn.Module):
    def __init__(self, nc=80, pretrained=True, verbose=True):
        super().__init__()
        # ... other code ...
        
        self.backbone = MobileNetV3BackboneDW(pretrained=pretrained)
        
        # UPDATE: Match backbone output channels
        self.neck = UltraLiteNeckDW(in_channels=[32, 64, 128])  # ← CHANGED from [24, 40, 160]
        
        # Neck output channels also need updating
        neck_out_channels = [32, 48, 64]  # Keep same or change
        
        self.head = Detect(nc=nc, ch=tuple(neck_out_channels))
```

---

## Modifying the Neck

### Example: Add Attention Modules to Neck

**Location:** `ultralytics/nn/modules/custom_mobilenet_blocks.py`

#### Step 1: Understand Current Neck Structure

```python
class UltraLiteNeckDW(nn.Module):
    def __init__(self, in_channels=[24, 40, 160]):
        super().__init__()
        # P3 path: 24 → 32
        self.reduce_p3 = DWConvCustom(in_channels[0], 32, k=1, s=1)
        
        # P4 path: 40 → 48
        self.reduce_p4 = DWConvCustom(in_channels[1], 48, k=1, s=1)
        self.sppf_p4 = SimSPPF(48, 48)
        
        # P5 path: 160 → 64
        self.reduce_p5 = DWConvCustom(in_channels[2], 64, k=1, s=1)
        self.transformer_p5 = P5Transformer(64, num_heads=2, num_layers=1)
        self.attention_p5 = CBAM_ChannelOnly(64)
```

#### Step 2: Add More Attention

```python
class UltraLiteNeckDW(nn.Module):
    def __init__(self, in_channels=[24, 40, 160]):
        super().__init__()
        # P3 path
        self.reduce_p3 = DWConvCustom(in_channels[0], 32, k=1, s=1)
        self.attention_p3 = CBAM_ChannelOnly(32)  # ← ADD attention to P3
        
        # P4 path
        self.reduce_p4 = DWConvCustom(in_channels[1], 48, k=1, s=1)
        self.sppf_p4 = SimSPPF(48, 48)
        self.attention_p4 = CBAM_ChannelOnly(48)  # ← ADD attention to P4
        
        # P5 path (already has attention)
        self.reduce_p5 = DWConvCustom(in_channels[2], 64, k=1, s=1)
        self.transformer_p5 = P5Transformer(64, num_heads=2, num_layers=1)
        self.attention_p5 = CBAM_ChannelOnly(64)
    
    def forward(self, features):
        p3, p4, p5 = features
        
        # P3 processing
        p3_out = self.reduce_p3(p3)
        p3_out = self.attention_p3(p3_out)  # ← ADD
        
        # P4 processing
        p4_out = self.reduce_p4(p4)
        p4_out = self.sppf_p4(p4_out)
        p4_out = self.attention_p4(p4_out)  # ← ADD
        
        # P5 processing (unchanged)
        p5_out = self.reduce_p5(p5)
        p5_out = self.transformer_p5(p5_out)
        p5_out = self.attention_p5(p5_out)
        
        return [p3_out, p4_out, p5_out]
```

#### Step 3: Change Output Channels

```python
class UltraLiteNeckDW(nn.Module):
    def __init__(self, in_channels=[24, 40, 160], out_channels=[64, 96, 128]):  # ← ADD parameter
        super().__init__()
        # P3 path: use out_channels[0]
        self.reduce_p3 = DWConvCustom(in_channels[0], out_channels[0], k=1, s=1)
        self.attention_p3 = CBAM_ChannelOnly(out_channels[0])
        
        # P4 path: use out_channels[1]
        self.reduce_p4 = DWConvCustom(in_channels[1], out_channels[1], k=1, s=1)
        self.sppf_p4 = SimSPPF(out_channels[1], out_channels[1])
        self.attention_p4 = CBAM_ChannelOnly(out_channels[1])
        
        # P5 path: use out_channels[2]
        self.reduce_p5 = DWConvCustom(in_channels[2], out_channels[2], k=1, s=1)
        self.transformer_p5 = P5Transformer(out_channels[2], num_heads=2, num_layers=1)
        self.attention_p5 = CBAM_ChannelOnly(out_channels[2])
```

**Then update the model:**

```python
# In ultralytics/nn/custom_models.py
class MobileNetV3YOLO(nn.Module):
    def __init__(self, nc=80, pretrained=True, verbose=True):
        # ... other code ...
        
        self.neck = UltraLiteNeckDW(
            in_channels=[24, 40, 160],
            out_channels=[64, 96, 128]  # ← NEW
        )
        
        # UPDATE: Head input channels must match neck output
        neck_out_channels = [64, 96, 128]  # ← CHANGED
        self.head = Detect(nc=nc, ch=tuple(neck_out_channels))
```

---

## Modifying the Head

### The Detection Head

**⚠️ NOTE:** The head is the standard YOLO Detect module. You usually **don't need to modify it**.

But if you want to use a different head:

#### Step 1: Import Different Head

```python
# In ultralytics/nn/custom_models.py
from ultralytics.nn.modules import Detect, Segment, Pose, OBB  # Different heads
```

#### Step 2: Change Head Type

```python
class MobileNetV3YOLO(nn.Module):
    def __init__(self, nc=80, pretrained=True, verbose=True):
        # ... other code ...
        
        # Option 1: Use Segment head for instance segmentation
        from ultralytics.nn.modules import Segment
        self.head = Segment(nc=nc, ch=tuple(neck_out_channels))
        
        # Option 2: Use OBB head for oriented bounding boxes
        from ultralytics.nn.modules import OBB
        self.head = OBB(nc=nc, ch=tuple(neck_out_channels))
        
        # Option 3: Use Pose head for keypoint detection
        from ultralytics.nn.modules import Pose
        self.head = Pose(nc=nc, ch=tuple(neck_out_channels), kpt_shape=(17, 3))
```

---

## Testing Your Changes

### Testing Workflow (CRITICAL!)

After ANY modification, follow this **exact sequence**:

#### Step 1: Quick Dimension Test

```python
# test_dimensions.py
import torch
from ultralytics import YOLO

print("Testing model dimensions...")

# Load model
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

# Test with dummy input
x = torch.randn(1, 3, 640, 640)

# Forward pass
model.model.eval()
with torch.no_grad():
    output = model.model(x)

print(f"✓ Input: {x.shape}")
print(f"✓ Output: {output[0].shape}")
print(f"✓ Feature maps: {len(output[1])}")
for i, feat in enumerate(output[1]):
    print(f"  P{i+3}: {feat.shape}")

print("\n✓ Dimensions look good!")
```

**Run it:**
```bash
python test_dimensions.py
```

**Expected output:**
```
✓ Input: torch.Size([1, 3, 640, 640])
✓ Output: torch.Size([1, 84, 8400])
✓ Feature maps: 3
  P3: torch.Size([1, 144, 80, 80])
  P4: torch.Size([1, 144, 40, 40])
  P5: torch.Size([1, 144, 20, 20])
```

#### Step 2: Loss Calculation Test

```python
# test_loss.py
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

print("Testing loss calculation...")

model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
model.model.args = get_cfg()

# Create dummy batch
batch = {
    'img': torch.randn(2, 3, 640, 640),
    'cls': torch.randint(0, 4, (10, 1)).float(),
    'bboxes': torch.rand(10, 4),
    'batch_idx': torch.cat([torch.zeros(5), torch.ones(5)]).long(),
}

model.model.train()
loss, loss_items = model.model(batch)

print(f"✓ Total loss: {loss.sum().item():.4f}")
print(f"✓ Box loss: {loss_items[0]:.4f}")
print(f"✓ Cls loss: {loss_items[1]:.4f}")
print(f"✓ DFL loss: {loss_items[2]:.4f}")
print("\n✓ Loss calculation works!")
```

#### Step 3: Backward Pass Test

```python
# test_backward.py
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

print("Testing backward pass...")

model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
model.model.args = get_cfg()

batch = {
    'img': torch.randn(2, 3, 640, 640),
    'cls': torch.randint(0, 4, (10, 1)).float(),
    'bboxes': torch.rand(10, 4),
    'batch_idx': torch.cat([torch.zeros(5), torch.ones(5)]).long(),
}

model.model.train()
model.model.zero_grad()

loss, loss_items = model.model(batch)
loss.sum().backward()

# Check gradients
grad_count = sum(1 for p in model.model.parameters() if p.grad is not None)
param_count = sum(1 for p in model.model.parameters())

print(f"✓ Parameters with gradients: {grad_count}/{param_count}")
print("✓ Backward pass works!")
```

#### Step 4: Full Training Test

Use the existing `quick_test.py`:

```bash
python quick_test.py
```

---

## Common Modification Patterns

### Pattern 1: Make Model Wider (More Channels)

**Goal:** Increase model capacity

```python
# Backbone: Increase output channels
self.conv_p3_dw = DWConvCustom(24, 48, k=3, s=1)   # 24 → 48
self.conv_p4_dw = DWConvCustom(40, 80, k=3, s=1)   # 40 → 80
self.conv_p5_dw = DWConvCustom(160, 256, k=3, s=1) # 160 → 256

# Neck: Update input and output channels
self.neck = UltraLiteNeckDW(in_channels=[48, 80, 256])

# In neck, increase output too
self.reduce_p3 = DWConvCustom(48, 64, k=1, s=1)   # 32 → 64
self.reduce_p4 = DWConvCustom(80, 96, k=1, s=1)   # 48 → 96
self.reduce_p5 = DWConvCustom(256, 128, k=1, s=1) # 64 → 128

# Head: Update input channels
self.head = Detect(nc=nc, ch=(64, 96, 128))
```

**Result:** ~2-3x more parameters, better accuracy, slower inference

### Pattern 2: Make Model Deeper (More Layers)

**Goal:** Learn more complex patterns

```python
class DeeperNeck(nn.Module):
    def __init__(self, in_channels=[24, 40, 160]):
        super().__init__()
        # P3 path - add 2 extra layers
        self.reduce_p3 = DWConvCustom(in_channels[0], 32, k=1, s=1)
        self.extra_p3_1 = DWConvCustom(32, 32, k=3, s=1)  # ← ADD
        self.extra_p3_2 = DWConvCustom(32, 32, k=3, s=1)  # ← ADD
        
        # Similar for P4 and P5
        # ...
    
    def forward(self, features):
        p3, p4, p5 = features
        
        p3_out = self.reduce_p3(p3)
        p3_out = self.extra_p3_1(p3_out)  # ← ADD
        p3_out = self.extra_p3_2(p3_out)  # ← ADD
        
        # ...
        return [p3_out, p4_out, p5_out]
```

**Result:** Better feature extraction, more parameters, slower

### Pattern 3: Add Skip Connections

**Goal:** Better gradient flow

```python
class NeckWithSkips(nn.Module):
    def forward(self, features):
        p3, p4, p5 = features
        
        # Process
        p3_out = self.reduce_p3(p3)
        p3_out = self.conv_p3(p3_out)
        
        # ADD: Skip connection
        p3_out = p3_out + self.reduce_p3(p3)  # ← Residual connection
        
        # Similar for P4, P5
        # ...
```

### Pattern 4: Use Different Backbones

**Goal:** Try ResNet, EfficientNet, etc.

```python
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models
        
        # Use ResNet50 instead of MobileNetV3
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layers
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # → P3
        self.layer2 = resnet.layer2  # → P4
        self.layer3 = resnet.layer3  # → P5
    
    def forward(self, x):
        x = self.stem(x)
        p3 = self.layer1(x)   # 256 channels
        p4 = self.layer2(p3)  # 512 channels
        p5 = self.layer3(p4)  # 1024 channels
        
        return [p3, p4, p5]

# Then update model
class CustomYOLO(nn.Module):
    def __init__(self, nc=80, pretrained=True):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.neck = UltraLiteNeckDW(in_channels=[256, 512, 1024])  # ← UPDATE
        # ...
```

---

## Troubleshooting

### Issue 1: Shape Mismatch Errors

**Error:**
```
RuntimeError: size mismatch, got input (4, 64, 80, 80), expected (4, 32, 80, 80)
```

**Solution:**
- Check that backbone output channels match neck input channels
- Check that neck output channels match head input channels
- Use dimension test (Step 1 in Testing)

### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size
model.train(batch=8)  # Instead of 16

# Or reduce image size
model.train(imgsz=512)  # Instead of 640

# Or reduce model width (fewer channels)
```

### Issue 3: Model Not Learning

**Symptoms:**
- Loss not decreasing
- mAP stays at 0

**Solution:**
1. Check learning rate: `lr0=0.001` (not too high)
2. Check data: Verify labels are correct
3. Check channels: Make sure all connections are correct
4. Try pretrained backbone: `pretrained=True`

### Issue 4: Gradient Issues

**Error:**
```
RuntimeError: one of the variables needed for gradient computation has been modified
```

**Solution:**
- Don't use in-place operations: `x = x + y` instead of `x += y`
- Use `.clone()` when needed
- Check for tensor reuse in forward pass

### Issue 5: Dimension Test Passes but Training Fails

**Problem:** Test works, but training crashes

**Common causes:**
1. Forgot to update `model.args` in model initialization
2. Batch dict keys missing ('img', 'cls', 'bboxes', 'batch_idx')
3. Loss function incompatible with head type

**Solution:**
```python
# Always check these in custom_models.py
class YourModel(nn.Module):
    def __init__(self, ...):
        # ...
        self.model = nn.ModuleList([...])  # ← Must have this
        self.args = None  # ← Must have this
        self.task = 'detect'  # ← Must have this
```

---

## Quick Reference: Modification Checklist

When modifying your model, follow this checklist:

- [ ] **1. Modify the module** (backbone/neck/head)
- [ ] **2. Update channel connections** (output → input matching)
- [ ] **3. Run dimension test** (`test_dimensions.py`)
- [ ] **4. Run loss test** (`test_loss.py`)
- [ ] **5. Run backward test** (`test_backward.py`)
- [ ] **6. Run quick test** (`quick_test.py`)
- [ ] **7. Try 1 epoch training** (use dummy data or real data)
- [ ] **8. Check model size** (`model.info()`)
- [ ] **9. Commit changes** (`git commit`)
- [ ] **10. Full training** (if everything works)

---

## Examples: Real Modifications

### Example 1: Add Dropout for Regularization

```python
class UltraLiteNeckDW(nn.Module):
    def __init__(self, in_channels=[24, 40, 160], dropout=0.1):  # ← ADD parameter
        super().__init__()
        self.reduce_p3 = DWConvCustom(in_channels[0], 32, k=1, s=1)
        self.dropout_p3 = nn.Dropout2d(dropout)  # ← ADD dropout
        
        # Similar for other paths
    
    def forward(self, features):
        p3, p4, p5 = features
        
        p3_out = self.reduce_p3(p3)
        p3_out = self.dropout_p3(p3_out)  # ← APPLY dropout
        
        # ...
```

### Example 2: Use Batch Normalization

```python
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k//2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)  # ← ADD BN
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))  # ← BN before ReLU

# Use in backbone/neck
self.reduce_p3 = ConvBNReLU(24, 32, k=1, s=1)
```

### Example 3: Multi-Scale Feature Fusion

```python
class FPNNeck(nn.Module):
    """Feature Pyramid Network style neck"""
    def __init__(self, in_channels=[24, 40, 160]):
        super().__init__()
        # Top-down pathway
        self.lat_p5 = nn.Conv2d(in_channels[2], 64, 1)
        self.lat_p4 = nn.Conv2d(in_channels[1], 64, 1)
        self.lat_p3 = nn.Conv2d(in_channels[0], 64, 1)
        
        # Smooth layers
        self.smooth_p3 = nn.Conv2d(64, 64, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(64, 64, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(64, 64, 3, padding=1)
    
    def forward(self, features):
        p3, p4, p5 = features
        
        # Top-down fusion
        p5_out = self.lat_p5(p5)
        
        p4_lat = self.lat_p4(p4)
        p5_up = F.interpolate(p5_out, size=p4.shape[-2:], mode='nearest')
        p4_out = p4_lat + p5_up
        
        p3_lat = self.lat_p3(p3)
        p4_up = F.interpolate(p4_out, size=p3.shape[-2:], mode='nearest')
        p3_out = p3_lat + p4_up
        
        # Smooth
        p3_out = self.smooth_p3(p3_out)
        p4_out = self.smooth_p4(p4_out)
        p5_out = self.smooth_p5(p5_out)
        
        return [p3_out, p4_out, p5_out]
```

---

## Tips for Safe Development

### 1. Always Version Control

```bash
# Before making changes
git add .
git commit -m "Working baseline"
git checkout -b experiment-wider-neck

# Make changes...

# If it works
git checkout main
git merge experiment-wider-neck

# If it doesn't work
git checkout main
git branch -D experiment-wider-neck
```

### 2. Keep a Working Baseline

Never delete your working code. Keep it in a separate file:

```
custom_mobilenet_blocks.py          ← Working version
custom_mobilenet_blocks_v2.py       ← Experiment
custom_mobilenet_blocks_resnet.py   ← Different backbone
```

### 3. Document Your Changes

Add comments:

```python
class MobileNetV3BackboneDW(nn.Module):
    """
    MobileNetV3 Small backbone with depthwise convolutions.
    
    Modifications from original:
    - Added extra DW conv layers (2024-12-05)
    - Changed P3 channels from 24 to 32 (2024-12-05)
    - Added residual connections (2024-12-06)
    """
```

### 4. Test Incrementally

Don't change everything at once:
- Change backbone → Test
- Change neck → Test
- Change head → Test

### 5. Monitor Model Size

```python
# After changes, check model size
model = YOLO('...')
model.info()

# Look at:
# - Total parameters (should not explode)
# - GFLOPs (computational cost)
# - Layers (complexity)
```

---

## Summary

You now know how to:

✅ Modify the backbone safely  
✅ Modify the neck safely  
✅ Change output channels  
✅ Add extra layers  
✅ Test your modifications  
✅ Debug common issues  
✅ Use the repo for any custom model  

**Remember:** Always test after each change, and use version control!

---

**Happy model development! 🚀**
