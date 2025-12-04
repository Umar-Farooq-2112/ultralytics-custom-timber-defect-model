# File Update Reference Guide

## Complete File-by-File Modification Guide

This guide shows **exactly which files to update** and **what content to change** for any custom model modification.

---

## 📋 Table of Contents

1. [Quick Reference Matrix](#quick-reference-matrix)
2. [File-by-File Details](#file-by-file-details)
3. [Modification Scenarios](#modification-scenarios)
4. [Update Checklist](#update-checklist)

---

## Quick Reference Matrix

| What You Want to Change | Files to Update | Order |
|-------------------------|-----------------|-------|
| **Add/modify backbone layers** | 1. `custom_mobilenet_blocks.py`<br>2. `custom_models.py` | 1→2 |
| **Add/modify neck layers** | 1. `custom_mobilenet_blocks.py`<br>2. `custom_models.py` | 1→2 |
| **Change backbone output channels** | 1. `custom_mobilenet_blocks.py`<br>2. `custom_models.py` (neck input)<br>3. `custom_models.py` (head input if needed) | 1→2→3 |
| **Change neck output channels** | 1. `custom_mobilenet_blocks.py`<br>2. `custom_models.py` (head input) | 1→2 |
| **Add new custom module** | 1. `custom_mobilenet_blocks.py`<br>2. `modules/__init__.py`<br>3. Use it in `custom_models.py` | 1→2→3 |
| **Change model name** | 1. Rename YAML file<br>2. `tasks.py` (parse logic)<br>3. Training scripts | 1→2→3 |
| **Use different head** | 1. `custom_models.py` only | 1 |
| **Change number of classes** | Training script (`nc=` parameter) | 1 |

---

## File-by-File Details

### File 1: `ultralytics/nn/modules/custom_mobilenet_blocks.py`

**Purpose:** Contains all custom module implementations (backbone, neck, individual blocks)

**When to Update:**
- Adding/removing layers in backbone or neck
- Changing channel dimensions
- Adding new custom blocks
- Modifying existing modules

**What to Update:**

#### Scenario A: Adding Layers to Backbone

```python
# LOCATION: class MobileNetV3BackboneDW
# LINE: ~190-210

class MobileNetV3BackboneDW(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # ... existing code ...
        
        # FIND THIS SECTION:
        self.conv_p3_dw = DWConvCustom(24, 24, k=3, s=1)
        self.conv_p4_dw = DWConvCustom(40, 40, k=3, s=1)
        self.conv_p5_dw = DWConvCustom(160, 160, k=3, s=1)
        
        # ADD NEW LAYERS HERE:
        self.extra_layer = DWConvCustom(160, 160, k=3, s=1)  # ← ADD
```

**Content Template:**
```python
# To add a new layer:
self.new_layer_name = DWConvCustom(in_channels, out_channels, k=kernel, s=stride)

# To change channels:
# OLD: self.conv_p3_dw = DWConvCustom(24, 24, k=3, s=1)
# NEW: self.conv_p3_dw = DWConvCustom(24, 32, k=3, s=1)  # 24→32 output
```

#### Scenario B: Modifying Neck

```python
# LOCATION: class UltraLiteNeckDW
# LINE: ~240-280

class UltraLiteNeckDW(nn.Module):
    def __init__(self, in_channels=[24, 40, 160]):
        super().__init__()
        
        # FIND THIS SECTION - P3 path:
        self.reduce_p3 = DWConvCustom(in_channels[0], 32, k=1, s=1)
        
        # ADD NEW MODULES:
        self.attention_p3 = CBAM_ChannelOnly(32)  # ← ADD
        
    def forward(self, features):
        p3, p4, p5 = features
        
        # FIND THIS SECTION - P3 processing:
        p3_out = self.reduce_p3(p3)
        
        # ADD NEW OPERATIONS:
        p3_out = self.attention_p3(p3_out)  # ← ADD
```

**Content Template:**
```python
# To add a module to neck:
# 1. In __init__:
self.new_module = ModuleName(channels, parameters)

# 2. In forward:
p3_out = self.new_module(p3_out)

# To change output channels:
# OLD: self.reduce_p3 = DWConvCustom(in_channels[0], 32, k=1, s=1)
# NEW: self.reduce_p3 = DWConvCustom(in_channels[0], 64, k=1, s=1)  # 32→64
```

#### Scenario C: Adding New Custom Block

```python
# LOCATION: Anywhere in file before backbone/neck classes
# LINE: After other block definitions (~50-150)

# ADD NEW CLASS:
class YourCustomBlock(nn.Module):
    """Your custom module description."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

---

### File 2: `ultralytics/nn/modules/__init__.py`

**Purpose:** Exports all modules to make them importable

**When to Update:**
- Added a new custom block/module
- Want to make it available for import

**What to Update:**

```python
# LOCATION: Imports section
# LINE: ~30-60 (after standard module imports)

# FIND THIS SECTION:
__all__ = (
    # ... existing imports ...
    "MobileNetV3BackboneDW",
    "UltraLiteNeckDW",
    "DWConvCustom",
    "ConvBNAct",
    "CBAM_ChannelOnly",
    "SimSPPF",
    "P5Transformer",
)

# ADD YOUR NEW MODULE:
__all__ = (
    # ... existing imports ...
    "YourCustomBlock",  # ← ADD
)
```

**Content Template:**
```python
# Pattern:
__all__ = (
    "ExistingModule1",
    "ExistingModule2",
    "YourNewModule",  # ← Add here (with comma!)
)
```

---

### File 3: `ultralytics/nn/custom_models.py`

**Purpose:** Main model class that connects backbone, neck, and head

**When to Update:**
- Changed backbone output channels
- Changed neck input/output channels  
- Changed model architecture
- Using different head type

**What to Update:**

#### Scenario A: Update Channel Connections

```python
# LOCATION: class MobileNetV3YOLO.__init__
# LINE: ~40-55

def __init__(self, nc=80, pretrained=True, verbose=True):
    # ... existing code ...
    
    # FIND THIS:
    self.backbone = MobileNetV3BackboneDW(pretrained=pretrained)
    self.neck = UltraLiteNeckDW(in_channels=[24, 40, 160])
    
    # If you changed backbone output channels from [24,40,160] to [32,64,128]:
    self.neck = UltraLiteNeckDW(in_channels=[32, 64, 128])  # ← UPDATE
    
    # FIND THIS:
    neck_out_channels = [32, 48, 64]
    
    # If you changed neck output channels to [64, 96, 128]:
    neck_out_channels = [64, 96, 128]  # ← UPDATE
    
    # This must match neck output:
    self.head = Detect(nc=nc, ch=tuple(neck_out_channels))
```

**Content Template:**
```python
# Always match these channels:
# backbone output → neck input
# neck output → head input

# Example flow:
# Backbone outputs: [24, 40, 160]
#     ↓
# Neck input: in_channels=[24, 40, 160]  ← MUST MATCH
# Neck outputs: [32, 48, 64]
#     ↓
# Head input: ch=(32, 48, 64)  ← MUST MATCH
```

#### Scenario B: Use Different Head

```python
# LOCATION: class MobileNetV3YOLO.__init__
# LINE: ~50-55

# FIND THIS:
from ultralytics.nn.modules import Detect

# CHANGE TO:
from ultralytics.nn.modules import Segment  # For segmentation
# OR
from ultralytics.nn.modules import OBB  # For oriented bounding boxes
# OR
from ultralytics.nn.modules import Pose  # For keypoint detection

# THEN UPDATE:
# OLD:
self.head = Detect(nc=nc, ch=tuple(neck_out_channels))

# NEW (Segmentation):
self.head = Segment(nc=nc, ch=tuple(neck_out_channels), nm=32)

# NEW (OBB):
self.head = OBB(nc=nc, ch=tuple(neck_out_channels))

# NEW (Pose):
self.head = Pose(nc=nc, ch=tuple(neck_out_channels), kpt_shape=(17, 3))
```

---

### File 4: `ultralytics/nn/tasks.py`

**Purpose:** Model parsing and loading logic

**When to Update:**
- Changing model identifier/name
- Adding new custom model type
- Modifying model selection logic

**What to Update:**

```python
# LOCATION: function parse_custom_model
# LINE: ~105-130

def parse_custom_model(cfg, ch=3, nc=80, verbose=True):
    """Parse custom model configurations."""
    from ultralytics.nn.custom_models import MobileNetV3YOLO
    
    # FIND THIS SECTION:
    if isinstance(cfg, dict):
        cfg_str = str(cfg.get('custom_model', '')).lower()
    elif isinstance(cfg, str):
        cfg_str = cfg.lower()
    else:
        return None
    
    # MODIFY DETECTION LOGIC:
    # OLD:
    if 'mobilenetv3' in cfg_str or 'mobilenet-v3' in cfg_str:
        return MobileNetV3YOLO(nc=nc, pretrained=True, verbose=verbose)
    
    # ADD NEW MODEL:
    if 'resnet-yolo' in cfg_str:
        return ResNetYOLO(nc=nc, pretrained=True, verbose=verbose)  # ← ADD
```

**Content Template:**
```python
# To add a new custom model:
# 1. Import it:
from ultralytics.nn.custom_models import YourModelClass

# 2. Add detection logic:
if 'your-model-name' in cfg_str:
    return YourModelClass(nc=nc, pretrained=True, verbose=verbose)
```

---

### File 5: `ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml`

**Purpose:** Model configuration file

**When to Update:**
- Rarely! This file just identifies the model
- Only if you want to add metadata or change task type

**What to Update:**

```yaml
# LOCATION: Entire file
# LINE: 1-10

# FIND THIS:
# Ultralytics MobileNetV3-YOLO Configuration
custom_model: mobilenetv3-yolo
task: detect  # detect, segment, classify, pose, obb

# Model parameters
nc: 80  # number of classes
depth_multiple: 1.0
width_multiple: 1.0

# MODIFY:
# To change task type:
task: segment  # ← CHANGE to segment/pose/obb

# To rename model:
custom_model: my-custom-model  # ← CHANGE (also update tasks.py!)
```

**Content Template:**
```yaml
# Minimal required content:
custom_model: your-model-name  # Must match check in tasks.py
task: detect  # or segment, pose, obb, classify
nc: 80  # Will be overridden by training script
```

---

### File 6: `ultralytics/models/yolo/detect/train.py`

**Purpose:** Detection trainer (sets model.args)

**When to Update:**
- Usually NO need to update (already configured)
- Only if adding custom trainer logic

**What to Update (if needed):**

```python
# LOCATION: class DetectionTrainer.get_model
# LINE: ~160-170

# CURRENT CODE (already working):
custom_model = parse_custom_model(cfg, ch=self.data.get("channels", 3), 
                                  nc=self.data["nc"], verbose=verbose and RANK == -1)
if custom_model is not None:
    custom_model.args = self.args  # ← Already sets args
    custom_model.pt_path = weights
    if weights:
        custom_model.load(weights)
    return custom_model

# ONLY modify if you need custom initialization:
if custom_model is not None:
    custom_model.args = self.args
    custom_model.pt_path = weights
    custom_model.your_custom_attr = some_value  # ← ADD custom attributes
```

---

### File 7: Training Scripts (Optional)

**Purpose:** Scripts to train your model

**When to Update:**
- Changing dataset path
- Adjusting hyperparameters
- Changing model path

**Files:**
- `train_custom_model.py`
- `quick_test.py`
- `test_training.py`

**What to Update:**

```python
# FIND THIS:
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

# CHANGE TO:
model = YOLO('ultralytics/cfg/models/custom/your-model.yaml')  # ← UPDATE

# FIND THIS:
model.train(
    data='coco8.yaml',
    epochs=100,
    # ...
)

# CHANGE TO:
model.train(
    data='your-dataset.yaml',  # ← UPDATE
    epochs=200,  # ← UPDATE
    batch=32,  # ← UPDATE
    # ...
)
```

---

## Modification Scenarios

### Scenario 1: Change Backbone from MobileNetV3 to ResNet50

**Files to Update:** 3 files

#### Step 1: Create New Backbone in `custom_mobilenet_blocks.py`

```python
# ADD at line ~200 (after MobileNetV3BackboneDW):

class ResNet50Backbone(nn.Module):
    """ResNet50 backbone for YOLO."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(pretrained=pretrained)
        
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
    
    def forward(self, x):
        x = self.stem(x)
        p3 = self.layer1(x)   # [B, 256, H/8, W/8]
        p4 = self.layer2(p3)  # [B, 512, H/16, W/16]
        p5 = self.layer3(p4)  # [B, 1024, H/32, W/32]
        return [p3, p4, p5]
```

#### Step 2: Export in `modules/__init__.py`

```python
# ADD to __all__ tuple:
__all__ = (
    # ... existing ...
    "ResNet50Backbone",  # ← ADD
)
```

#### Step 3: Use in `custom_models.py`

```python
# CHANGE imports:
from ultralytics.nn.modules import ResNet50Backbone  # ← ADD

# CHANGE in __init__:
# OLD:
self.backbone = MobileNetV3BackboneDW(pretrained=pretrained)
self.neck = UltraLiteNeckDW(in_channels=[24, 40, 160])

# NEW:
self.backbone = ResNet50Backbone(pretrained=pretrained)
self.neck = UltraLiteNeckDW(in_channels=[256, 512, 1024])  # ← UPDATE channels!
```

---

### Scenario 2: Add Attention to All Neck Levels

**Files to Update:** 1 file

#### Update `custom_mobilenet_blocks.py`

```python
# LOCATION: class UltraLiteNeckDW.__init__
# LINE: ~245

# ADD after reduce operations:
self.reduce_p3 = DWConvCustom(in_channels[0], 32, k=1, s=1)
self.attention_p3 = CBAM_ChannelOnly(32)  # ← ADD

self.reduce_p4 = DWConvCustom(in_channels[1], 48, k=1, s=1)
self.attention_p4 = CBAM_ChannelOnly(48)  # ← ADD

# (P5 already has attention)

# LOCATION: class UltraLiteNeckDW.forward
# LINE: ~265

# UPDATE forward pass:
p3_out = self.reduce_p3(p3)
p3_out = self.attention_p3(p3_out)  # ← ADD

p4_out = self.reduce_p4(p4)
p4_out = self.sppf_p4(p4_out)
p4_out = self.attention_p4(p4_out)  # ← ADD
```

---

### Scenario 3: Increase Model Width (More Channels)

**Files to Update:** 2 files

#### Step 1: Update Backbone Output in `custom_mobilenet_blocks.py`

```python
# LOCATION: class MobileNetV3BackboneDW.__init__
# LINE: ~195

# CHANGE output channels:
# OLD:
self.conv_p3_dw = DWConvCustom(24, 24, k=3, s=1)
self.conv_p4_dw = DWConvCustom(40, 40, k=3, s=1)
self.conv_p5_dw = DWConvCustom(160, 160, k=3, s=1)

# NEW (double the channels):
self.conv_p3_dw = DWConvCustom(24, 48, k=3, s=1)   # 24 → 48
self.conv_p4_dw = DWConvCustom(40, 80, k=3, s=1)   # 40 → 80
self.conv_p5_dw = DWConvCustom(160, 256, k=3, s=1) # 160 → 256
```

#### Step 2: Update Neck and Head in `custom_models.py`

```python
# LOCATION: class MobileNetV3YOLO.__init__
# LINE: ~45

# UPDATE neck input channels:
# OLD:
self.neck = UltraLiteNeckDW(in_channels=[24, 40, 160])

# NEW:
self.neck = UltraLiteNeckDW(in_channels=[48, 80, 256])  # ← MATCH backbone output

# ALSO UPDATE neck output if desired:
# OLD:
neck_out_channels = [32, 48, 64]

# NEW (double):
neck_out_channels = [64, 96, 128]

# Head automatically uses new channels (no change needed)
```

Then update neck internals in `custom_mobilenet_blocks.py`:

```python
# LOCATION: class UltraLiteNeckDW.__init__
# LINE: ~245

# UPDATE to match new output:
self.reduce_p3 = DWConvCustom(in_channels[0], 64, k=1, s=1)   # 32 → 64
self.reduce_p4 = DWConvCustom(in_channels[1], 96, k=1, s=1)   # 48 → 96
self.reduce_p5 = DWConvCustom(in_channels[2], 128, k=1, s=1)  # 64 → 128
```

---

## Update Checklist

### Before Making Changes

- [ ] Backup current working code (`git commit`)
- [ ] Create new branch (`git checkout -b experiment`)
- [ ] Document what you're changing

### During Changes

- [ ] Update module definition in `custom_mobilenet_blocks.py`
- [ ] Export new modules in `modules/__init__.py` (if added new ones)
- [ ] Update channel connections in `custom_models.py`
- [ ] Update YAML config if needed
- [ ] Update training scripts if needed

### After Changes

- [ ] Run dimension test
- [ ] Run loss calculation test
- [ ] Run backward pass test
- [ ] Check model.info() for parameter count
- [ ] Try 1 epoch training on dummy data
- [ ] Commit if successful

### Channel Connection Checklist

- [ ] Backbone output channels match neck input channels
- [ ] Neck output channels match head input channels
- [ ] All convolution channel dimensions are compatible
- [ ] No shape mismatches in forward pass

---

## Quick Command Reference

```bash
# Test dimensions
python -c "
import torch
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
x = torch.randn(1, 3, 640, 640)
model.model.eval()
with torch.no_grad():
    out = model.model(x)
print(f'Output: {out[0].shape}')
print(f'Features: {[f.shape for f in out[1]]}')
"

# Check model size
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
model.model.info()
"

# Quick loss test
python quick_test.py
```

---

## Summary: Files You'll Most Commonly Edit

| File | Frequency | Purpose |
|------|-----------|---------|
| `custom_mobilenet_blocks.py` | ⭐⭐⭐⭐⭐ Very Often | Module implementations |
| `custom_models.py` | ⭐⭐⭐⭐ Often | Channel connections |
| `modules/__init__.py` | ⭐⭐ Sometimes | Export new modules |
| `tasks.py` | ⭐ Rarely | Add new model types |
| `train.py` | ⭐ Rarely | Already configured |
| `*.yaml` | ⭐ Rarely | Model metadata |
| Training scripts | ⭐⭐⭐ Often | Hyperparameters |

**Most common workflow:**
1. Edit `custom_mobilenet_blocks.py` (add/modify modules)
2. Edit `custom_models.py` (update channels)
3. Test with `quick_test.py`
4. Done! ✅

---

**Remember:** Always match channel dimensions between connected layers!
