# Quick Start Guide: MobileNetV3-YOLO Integration

## ✅ What's Been Created

Your custom MobileNetV3-YOLO model is now fully implemented with:

1. **Custom Modules** (`ultralytics/nn/modules/custom_mobilenet_blocks.py`)
   - ✅ DWConvCustom
   - ✅ ConvBNAct
   - ✅ CBAM_ChannelOnly
   - ✅ SimSPPF
   - ✅ P5Transformer
   - ✅ MobileNetV3BackboneDW
   - ✅ UltraLiteNeckDW

2. **Model Class** (`ultralytics/nn/custom_models.py`)
   - ✅ MobileNetV3YOLO

3. **Configuration** (`ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml`)

4. **Training Script** (`train_mobilenetv3_yolo.py`)

5. **Documentation** 
   - ✅ MOBILENETV3_YOLO_README.md
   - ✅ This guide

## 🚀 Usage

### Step 1: Verify Installation

```python
# Test imports
from ultralytics.nn.custom_models import MobileNetV3YOLO
from ultralytics.nn.modules import (
    MobileNetV3BackboneDW,
    UltraLiteNeckDW,
    DWConvCustom
)

print("✓ All imports successful!")
```

### Step 2: Build and Test Model

```bash
# Run the test script
python train_mobilenetv3_yolo.py
```

Expected output:
```
============================================================
Testing MobileNetV3-YOLO Model Build
============================================================

 Testing forward pass...
✓ Forward pass successful!
  Input shape: torch.Size([1, 3, 640, 640])
  Output 0 shape: torch.Size([1, 3, 80, 80, 85])
  Output 1 shape: torch.Size([1, 3, 40, 40, 85])
  Output 2 shape: torch.Size([1, 3, 20, 20, 85])

------------------------------------------------------------
Model Information:
------------------------------------------------------------
...model summary...
```

### Step 3: Inference Test

```python
import torch
from ultralytics.nn.custom_models import MobileNetV3YOLO

# Create model
model = MobileNetV3YOLO(nc=80, pretrained=True)
model.eval()

# Load an image (example with random tensor)
img = torch.randn(1, 3, 640, 640)

# Run inference
with torch.no_grad():
    predictions = model(img)

print("✓ Inference successful!")
for i, pred in enumerate(predictions):
    print(f"  Scale {i}: {pred.shape}")
```

## 🎓 Training Integration Options

### Option A: Quick PyTorch Training (Simple)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics.nn.custom_models import MobileNetV3YOLO

# Create model
model = MobileNetV3YOLO(nc=80, pretrained=True).cuda()

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Your data loader
train_loader = DataLoader(...)  # Your dataset

# Training loop
for epoch in range(100):
    for images, targets in train_loader:
        images = images.cuda()
        
        # Forward
        outputs = model(images)
        
        # Compute loss (you need to implement this)
        loss = compute_yolo_loss(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item()}")
```

### Option B: YOLO Trainer Integration (Recommended)

Create `ultralytics/models/mobilenetv3/train.py`:

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.custom_models import MobileNetV3YOLO
import torch


class MobileNetV3YOLOTrainer(DetectionTrainer):
    """Custom trainer for MobileNetV3-YOLO."""
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load MobileNetV3-YOLO model."""
        model = MobileNetV3YOLO(
            nc=self.data.get('nc', 80),
            pretrained=weights is None,  # Use pretrained if no weights
            verbose=verbose
        )
        
        if weights:
            checkpoint = torch.load(weights, map_location='cpu')
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'].state_dict())
            else:
                model.load_state_dict(checkpoint)
        
        return model


# Usage
if __name__ == '__main__':
    trainer = MobileNetV3YOLOTrainer(
        overrides={
            'data': 'coco8.yaml',
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'device': 0,
            'project': 'runs/mobilenetv3-yolo',
            'name': 'train'
        }
    )
    trainer.train()
```

Then train:
```bash
python ultralytics/models/mobilenetv3/train.py
```

### Option C: Standard YOLO API (Future Enhancement)

To make it work with standard YOLO API, add to `ultralytics/models/__init__.py`:

```python
from ultralytics.nn.custom_models import MobileNetV3YOLO

class MobileNetV3YOLOWrapper:
    """Wrapper to make MobileNetV3YOLO compatible with YOLO API."""
    
    def __init__(self, model='mobilenetv3-yolo', task='detect'):
        self.model = MobileNetV3YOLO(nc=80, pretrained=True)
        self.task = task
    
    def train(self, **kwargs):
        from ultralytics.models.mobilenetv3.train import MobileNetV3YOLOTrainer
        trainer = MobileNetV3YOLOTrainer(overrides=kwargs)
        return trainer.train()
    
    def predict(self, source, **kwargs):
        # Implement prediction
        pass
    
    def val(self, **kwargs):
        # Implement validation
        pass
```

## 📊 Architecture Verification

### Check Channel Flow

```python
import torch
from ultralytics.nn.custom_models import MobileNetV3YOLO

model = MobileNetV3YOLO(nc=80, pretrained=True)

# Hook to capture intermediate outputs
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

# Register hooks
model.backbone.stage1.register_forward_hook(get_activation('backbone_p3'))
model.backbone.conv_p4_dw.register_forward_hook(get_activation('backbone_p4'))
model.backbone.conv_p5_dw.register_forward_hook(get_activation('backbone_p5'))

model.neck.out3.register_forward_hook(get_activation('neck_p3'))
model.neck.out4.register_forward_hook(get_activation('neck_p4'))
model.neck.out5.register_forward_hook(get_activation('neck_p5'))

# Forward pass
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    _ = model(x)

# Print shapes
print("Channel Flow Verification:")
print("-" * 50)
print(f"Backbone P3: {activations['backbone_p3'].shape} (expected: [1, 24, H, W])")
print(f"Backbone P4: {activations['backbone_p4'].shape} (expected: [1, 40, H, W])")
print(f"Backbone P5: {activations['backbone_p5'].shape} (expected: [1, 160, H, W])")
print(f"Neck P3:     {activations['neck_p3'].shape} (expected: [1, 32, H, W])")
print(f"Neck P4:     {activations['neck_p4'].shape} (expected: [1, 48, H, W])")
print(f"Neck P5:     {activations['neck_p5'].shape} (expected: [1, 64, H, W])")
```

## 🔍 Model Analysis

### Count Parameters

```python
from ultralytics.nn.custom_models import MobileNetV3YOLO

model = MobileNetV3YOLO(nc=80)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Model Size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
```

### Measure Inference Speed

```python
import torch
import time
from ultralytics.nn.custom_models import MobileNetV3YOLO

model = MobileNetV3YOLO(nc=80).cuda().eval()
x = torch.randn(1, 3, 640, 640).cuda()

# Warmup
for _ in range(50):
    with torch.no_grad():
        _ = model(x)

# Benchmark
torch.cuda.synchronize()
start = time.time()

iterations = 100
for _ in range(iterations):
    with torch.no_grad():
        _ = model(x)

torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / iterations
fps = 1 / avg_time

print(f"Average Inference Time: {avg_time*1000:.2f} ms")
print(f"FPS: {fps:.2f}")
```

## 📝 Dataset Preparation

Your model expects YOLO format datasets:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt
```

Dataset YAML (`dataset.yaml`):
```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 80  # number of classes
names: ['person', 'bicycle', 'car', ...]  # class names
```

## 🎯 Next Steps

1. **Test the model**: Run `python train_mobilenetv3_yolo.py`

2. **Prepare your dataset**: Convert to YOLO format

3. **Choose training method**: 
   - Option A (Simple): Direct PyTorch
   - Option B (Recommended): YOLO Trainer integration
   
4. **Train the model**: Start with small dataset (coco8)

5. **Evaluate**: Check mAP, precision, recall

6. **Export**: Convert to ONNX/TorchScript for deployment

## ❓ FAQ

**Q: Can I use this with the standard `YOLO()` class?**  
A: Not directly. You need to use `MobileNetV3YOLO()` class or create a wrapper.

**Q: How do I change the number of classes?**  
A: Pass `nc` parameter: `MobileNetV3YOLO(nc=10)`

**Q: Can I use different input sizes?**  
A: Yes, the model supports dynamic input sizes: `model(torch.randn(1, 3, 512, 512))`

**Q: How do I freeze the backbone?**  
A:
```python
model = MobileNetV3YOLO(nc=80)
for param in model.backbone.parameters():
    param.requires_grad = False
```

**Q: Can I modify the architecture?**  
A: Yes! Edit `custom_mobilenet_blocks.py` and adjust channel dimensions in the classes.

## 🐛 Common Issues

### Issue 1: Import Error
```
ModuleNotFoundError: No module named 'ultralytics.nn.custom_models'
```
**Solution**: Make sure files are in correct locations

### Issue 2: Shape Mismatch
```
RuntimeError: size mismatch
```
**Solution**: Check channel dimensions match throughout pipeline

### Issue 3: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or image size

## 📞 Support

For issues or questions:
1. Check this guide
2. Review `MOBILENETV3_YOLO_README.md`
3. Inspect `train_mobilenetv3_yolo.py`
4. Review Ultralytics documentation

---

**Your custom MobileNetV3-YOLO model is ready to use! 🚀**
