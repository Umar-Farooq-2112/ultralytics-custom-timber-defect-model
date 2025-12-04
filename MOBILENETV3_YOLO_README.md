# MobileNetV3-YOLO Custom Model

A lightweight custom YOLO model combining **MobileNetV3 Small backbone** with an **ultra-lightweight neck** and **YOLOv8n detection head**.

## 🏗️ Architecture Overview

```
Input (640x640x3)
      ↓
┌─────────────────────────────────────┐
│  MobileNetV3 Small Backbone         │
│  ├─ Stage1 [:3]   → P3 (24ch)       │
│  ├─ Stage2 [3:7]  → P4 (40ch)       │
│  └─ Stage3 [7:]   → P5 (576→160ch)  │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Ultra-Lightweight Neck              │
│  ├─ P3: CBAM → DWConv → 32ch        │
│  ├─ P4: CBAM → DWConv → 48ch        │
│  └─ P5: SPPF → Transformer → 64ch   │
│     └─ Features: Attention + Fusion │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  YOLOv8n Detection Head              │
│  ├─ cv2: Bbox regression            │
│  ├─ cv3: Classification              │
│  └─ DFL: Distribution Focal Loss    │
└─────────────────────────────────────┘
      ↓
   Predictions
```

## 📊 Model Characteristics

| Feature | Value |
|---------|-------|
| **Parameters** | ~1.5M (vs YOLOv8n ~3M) |
| **Backbone** | MobileNetV3 Small (pretrained) |
| **Neck** | Custom ultra-lightweight |
| **Head** | YOLOv8n Detect |
| **Input Size** | 640x640 (configurable) |
| **Output Scales** | 3 (P3/8, P4/16, P5/32) |
| **Channels** | P3=32, P4=48, P5=64 |

## 🚀 Quick Start

### 1. Test Model Build

```python
from ultralytics.nn.custom_models import MobileNetV3YOLO
import torch

# Create model
model = MobileNetV3YOLO(nc=80, pretrained=True, verbose=True)

# Test forward pass
x = torch.randn(1, 3, 640, 640)
outputs = model(x)

print(f"Input: {x.shape}")
print(f"Output: {[o.shape for o in outputs]}")
```

### 2. Run Test Script

```bash
python train_mobilenetv3_yolo.py
```

This will:
- ✅ Build the model
- ✅ Test forward pass
- ✅ Display model info
- ✅ Run inference example

### 3. Model Information

```python
model = MobileNetV3YOLO(nc=80, pretrained=True)
model.info(detailed=True)
```

## 📝 Architecture Details

### Backbone: MobileNetV3 Small

```python
MobileNetV3BackboneDW(pretrained=True)
├─ stage1: feats[:3]   → 24 channels  (stride 8)
├─ stage2: feats[3:7]  → 40 channels  (stride 16)
│  └─ DWConv(40→40)
└─ stage3: feats[7:]   → 576 channels (stride 32)
   └─ DWConv(576→160)
```

**Key Features:**
- Pretrained on ImageNet
- Efficient depthwise separable convolutions
- Optimized for mobile/edge devices
- Output: Multi-scale features [P3, P4, P5]

### Neck: UltraLiteNeckDW

```python
UltraLiteNeckDW(in_channels=[24, 40, 160])
├─ P3 path:
│  └─ CBAM → DWConv(24→32)
├─ P4 path:
│  └─ CBAM → DWConv(40→48) → Fusion
└─ P5 path:
   └─ SPPF → Transformer(160→48) → CBAM → DWConv(48→64)
```

**Key Components:**
1. **CBAM_ChannelOnly**: Channel attention (reduction=8)
2. **SimSPPF**: Spatial pyramid pooling (4x pool, concat)
3. **P5Transformer**: 2-layer transformer (embed_dim=48, 1 head)
4. **DWConvCustom**: Depthwise separable convolutions
5. **Feature Fusion**: Adaptive pooling + concatenation

### Head: YOLOv8n Detect

```python
Detect(nc=80, ch=[32, 48, 64])
├─ cv2: Bbox regression (DFL)
└─ cv3: Classification (BCE)
```

**Detection Outputs:**
- P3 (32ch): Small objects (stride 8)
- P4 (48ch): Medium objects (stride 16)
- P5 (64ch): Large objects (stride 32)

## 🔧 Custom Modules

All custom modules are in `ultralytics/nn/modules/custom_mobilenet_blocks.py`:

### DWConvCustom
```python
DWConvCustom(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
```
Depthwise separable convolution: depthwise 3x3 + pointwise 1x1

### ConvBNAct
```python
ConvBNAct(in_ch, out_ch, k=1, stride=1, padding=0)
```
Standard convolution + BatchNorm + SiLU activation

### CBAM_ChannelOnly
```python
CBAM_ChannelOnly(channels, reduction=8)
```
Lightweight channel attention mechanism

### SimSPPF
```python
SimSPPF(c1, c2, k=5)
```
Simplified Spatial Pyramid Pooling Fast

### P5Transformer
```python
P5Transformer(in_channels, embed_dim=48, ff_dim=96, num_layers=2)
```
Transformer encoder for P5 features

## 📦 Files Structure

```
ultralytics/
├── nn/
│   ├── custom_models.py                      # MobileNetV3YOLO class
│   └── modules/
│       ├── custom_mobilenet_blocks.py        # Custom modules
│       └── __init__.py                       # Module exports
├── cfg/
│   └── models/
│       └── custom/
│           └── mobilenetv3-yolo.yaml         # Model config
└── train_mobilenetv3_yolo.py                 # Training script
```

## 🎯 Training (Integration Required)

The model is currently set up for testing. For full training:

### Option 1: Direct PyTorch Training

```python
import torch
from ultralytics.nn.custom_models import MobileNetV3YOLO

# Create model
model = MobileNetV3YOLO(nc=80, pretrained=True)

# Your training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# ... training code ...
```

### Option 2: Integrate with YOLO Trainer (Recommended)

Create a custom trainer by extending `DetectionTrainer`:

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.custom_models import MobileNetV3YOLO

class MobileNetV3Trainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Override to use custom model."""
        model = MobileNetV3YOLO(
            nc=self.data['nc'],
            pretrained=True,
            verbose=verbose
        )
        if weights:
            model.load_state_dict(torch.load(weights)['model'])
        return model

# Train
trainer = MobileNetV3Trainer(overrides={'data': 'coco8.yaml', 'epochs': 100})
trainer.train()
```

### Option 3: Convert to Standard YAML Format

Create a pure YAML-based model (requires implementing modules in `parse_model()`):

```yaml
# Future implementation
backbone:
  - [-1, 1, MobileNetV3BackboneDW, [True]]
neck:
  - [-1, 1, UltraLiteNeckDW, [[24, 40, 160]]]
head:
  - [[0, 1, 2], 1, Detect, [nc]]
```

## 🔬 Testing & Validation

### Test Model Building
```python
python train_mobilenetv3_yolo.py
```

### Check Model Info
```python
from ultralytics.nn.custom_models import MobileNetV3YOLO

model = MobileNetV3YOLO(nc=80, pretrained=True)
model.info(detailed=True, verbose=True)
```

### Profile Performance
```python
import torch
from ultralytics.nn.custom_models import MobileNetV3YOLO

model = MobileNetV3YOLO(nc=80).eval()
x = torch.randn(1, 3, 640, 640)

# Warmup
for _ in range(10):
    _ = model(x)

# Benchmark
import time
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(x)
end = time.time()

fps = 100 / (end - start)
print(f"FPS: {fps:.2f}")
```

## 🐛 Troubleshooting

### Import Errors
```python
# Make sure custom modules are imported
from ultralytics.nn.modules import (
    MobileNetV3BackboneDW,
    UltraLiteNeckDW,
    DWConvCustom,
    CBAM_ChannelOnly,
    SimSPPF,
    P5Transformer
)
```

### Shape Mismatches
The model expects specific channel dimensions:
- Backbone outputs: [24, 40, 160]
- Neck outputs: [32, 48, 64]
- Head inputs: [32, 48, 64]

If you modify channels, ensure compatibility throughout the pipeline.

### Missing torchvision
```bash
pip install torchvision
```

The MobileNetV3 backbone requires torchvision for pretrained weights.

## 📈 Expected Performance

| Metric | Value (Estimated) |
|--------|-------------------|
| **Parameters** | ~1.5M |
| **GFLOPs** | ~2.5 |
| **Inference (GPU)** | ~200 FPS |
| **Inference (CPU)** | ~30 FPS |
| **mAP@0.5** | TBD (needs training) |
| **mAP@0.5:0.95** | TBD (needs training) |

*Note: Actual performance depends on dataset and training configuration*

## 🔄 Next Steps

1. ✅ **Model Building** - Complete
2. ✅ **Module Integration** - Complete
3. ✅ **Forward Pass** - Complete
4. ⏳ **Training Pipeline** - Requires DetectionTrainer integration
5. ⏳ **Dataset Preparation** - User-specific
6. ⏳ **Training** - User-specific
7. ⏳ **Evaluation** - After training
8. ⏳ **Export** - ONNX/TorchScript ready

## 📚 References

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [CBAM](https://arxiv.org/abs/1807.06521)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)

## 📄 License

AGPL-3.0 License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests or open issues.

---

**Built with ❤️ using Ultralytics YOLO framework**
