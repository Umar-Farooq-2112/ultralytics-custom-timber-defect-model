# Complete Guide to Creating Custom YOLO Models

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Step 1: Create Custom Modules](#step-1-create-custom-modules)
3. [Step 2: Register Modules](#step-2-register-modules)
4. [Step 3: Create Model YAML Configuration](#step-3-create-model-yaml-configuration)
5. [Step 4: Train Your Custom Model](#step-4-train-your-custom-model)
6. [Advanced Customizations](#advanced-customizations)
7. [Complete Examples](#complete-examples)

---

## Architecture Overview

A YOLO model consists of three main components:

```
Input Image (640x640x3)
        ↓
┌───────────────────┐
│   BACKBONE        │  ← Extracts features at multiple scales
│   (Custom)        │    Outputs: P3, P4, P5 (8x, 16x, 32x downsampling)
└───────────────────┘
        ↓
┌───────────────────┐
│   NECK            │  ← Fuses features from different scales
│   (Custom)        │    Uses FPN + PAN architecture
└───────────────────┘
        ↓
┌───────────────────┐
│   HEAD            │  ← Predicts boxes, classes, masks, etc.
│   (Detect/Segment)│    Task-specific output layer
└───────────────────┘
```

**Key Concepts:**
- **Backbone**: Feature extractor (CSPDarknet, ResNet, EfficientNet, etc.)
- **Neck**: Feature fusion network (FPN, PAN, BiFPN, etc.)
- **Head**: Task-specific prediction layer (already implemented in Ultralytics)

---

## Step 1: Create Custom Modules

### 1.1 Understanding Module Structure

All custom modules should be created in `ultralytics/nn/modules/` directory.

**Basic Module Template:**
```python
import torch
import torch.nn as nn
from .conv import Conv

class CustomBlock(nn.Module):
    """
    Custom block description.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size
        s (int): Stride
    """
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, k, s)
        self.cv2 = Conv(c2, c2, k, s)
    
    def forward(self, x):
        """Forward pass."""
        return self.cv2(self.cv1(x))
```

### 1.2 Create Custom Backbone Block

Create file: `ultralytics/nn/modules/custom_blocks.py`

```python
# ultralytics/nn/modules/custom_blocks.py
"""Custom modules for YOLO models."""

import torch
import torch.nn as nn
from .conv import Conv, DWConv, autopad
from .block import Bottleneck

__all__ = (
    "CustomCSPBlock",
    "CustomAttentionBlock",
    "CustomResidualBlock",
    "CustomDownsample",
)


class CustomCSPBlock(nn.Module):
    """
    Custom CSP (Cross Stage Partial) Block.
    
    This is an example of creating a custom backbone block that processes
    features through split-transform-merge architecture.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of bottlenecks
        shortcut (bool): Use shortcut connection
        g (int): Groups for convolution
        e (float): Expansion ratio
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # output conv
        self.m = nn.Sequential(*(
            Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        ))
    
    def forward(self, x):
        """Forward pass through CSP block."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class CustomAttentionBlock(nn.Module):
    """
    Custom Attention Block with channel and spatial attention.
    
    Combines channel attention (squeeze-excitation) with spatial attention
    for enhanced feature representation.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        reduction (int): Channel reduction ratio
    """
    def __init__(self, c1, c2, reduction=16):
        super().__init__()
        self.conv = Conv(c1, c2, 3, 1)
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2 // reduction, c2, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(c2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass with attention mechanisms."""
        x = self.conv(x)
        
        # Apply channel attention
        ca = self.channel_attn(x)
        x = x * ca
        
        # Apply spatial attention
        sa = self.spatial_attn(x)
        x = x * sa
        
        return x


class CustomResidualBlock(nn.Module):
    """
    Custom Residual Block with multiple convolutions.
    
    Implements a residual connection with optional bottleneck structure
    for efficient feature extraction.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of convolutions
        e (float): Expansion ratio
    """
    def __init__(self, c1, c2, n=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*(Conv(c_, c_, 3, 1) for _ in range(n)))
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = c1 == c2
    
    def forward(self, x):
        """Forward pass with residual connection."""
        out = self.cv2(self.m(self.cv1(x)))
        return x + out if self.add else out


class CustomDownsample(nn.Module):
    """
    Custom Downsampling Block.
    
    Efficient downsampling using parallel paths with different kernel sizes,
    then concatenating results.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
    """
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 3, 2)  # stride=2 for downsampling
        self.cv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv(c1, c_, 1, 1)
        )
    
    def forward(self, x):
        """Forward pass combining conv and pooling paths."""
        return torch.cat((self.cv1(x), self.cv2(x)), 1)


class CustomNeckBlock(nn.Module):
    """
    Custom Neck Block for feature fusion.
    
    This block is designed for use in the neck portion of the network,
    fusing features from different scales.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of feature fusion layers
    """
    def __init__(self, c1, c2, n=2):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*(Conv(c_, c_, 3, 1) for _ in range(n)))
        self.cv3 = Conv(c2, c2, 1, 1)
    
    def forward(self, x):
        """Forward pass for neck feature fusion."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    
    Lightweight channel attention without dimensionality reduction.
    
    Args:
        channels (int): Number of input channels
        gamma (int): Kernel size for 1D convolution
        b (int): Constant for adaptive kernel size
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass with efficient channel attention."""
        # Global average pooling
        y = self.avg_pool(x)
        # 1D convolution along channel dimension
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        # Apply attention weights
        return x * self.sigmoid(y)


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (MobileNetV2-style).
    
    Expansion -> Depthwise -> Projection architecture.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        s (int): Stride
        e (float): Expansion ratio
    """
    def __init__(self, c1, c2, s=1, e=6):
        super().__init__()
        c_ = int(c1 * e)
        self.use_res = s == 1 and c1 == c2
        
        self.conv = nn.Sequential(
            # Expansion
            Conv(c1, c_, 1, 1) if e != 1 else nn.Identity(),
            # Depthwise
            DWConv(c_, c_, 3, s),
            # Projection
            nn.Conv2d(c_, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2)
        )
    
    def forward(self, x):
        """Forward pass with inverted residual."""
        return x + self.conv(x) if self.use_res else self.conv(x)
```

### 1.3 Register Custom Modules

Edit `ultralytics/nn/modules/__init__.py` to export your custom modules:

```python
# Add to ultralytics/nn/modules/__init__.py

from .custom_blocks import (
    CustomCSPBlock,
    CustomAttentionBlock,
    CustomResidualBlock,
    CustomDownsample,
    CustomNeckBlock,
    EfficientChannelAttention,
    InvertedResidual,
)

__all__ = (
    # ... existing modules ...
    "CustomCSPBlock",
    "CustomAttentionBlock",
    "CustomResidualBlock",
    "CustomDownsample",
    "CustomNeckBlock",
    "EfficientChannelAttention",
    "InvertedResidual",
)
```

---

## Step 2: Register Modules in Tasks

Edit `ultralytics/nn/tasks.py` to make modules available in YAML configs.

Find the `parse_model()` function and add your modules to the imports:

```python
# In ultralytics/nn/tasks.py

from ultralytics.nn.modules import (
    # ... existing imports ...
    CustomCSPBlock,
    CustomAttentionBlock,
    CustomResidualBlock,
    CustomDownsample,
    CustomNeckBlock,
    EfficientChannelAttention,
    InvertedResidual,
)
```

Then add them to the appropriate module sets in `parse_model()`:

```python
# In parse_model() function, add to base_modules:
base_modules = frozenset(
    {
        # ... existing modules ...
        CustomCSPBlock,
        CustomAttentionBlock,
        CustomResidualBlock,
        CustomDownsample,
        CustomNeckBlock,
        EfficientChannelAttention,
        InvertedResidual,
    }
)

# Add to repeat_modules if they support 'n' (number of repeats):
repeat_modules = frozenset(
    {
        # ... existing modules ...
        CustomCSPBlock,
        CustomResidualBlock,
        CustomNeckBlock,
    }
)
```

---

## Step 3: Create Model YAML Configuration

### 3.1 Understanding YAML Structure

```yaml
# Model configuration structure

# Metadata
nc: 80  # number of classes
scales: # model variants (n/s/m/l/x)
  n: [depth_multiple, width_multiple, max_channels]

# Architecture
backbone:
  # [from, number, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2

head:
  # [from, number, module, args]
  - [[16, 19, 22], 1, Detect, [nc]]  # Detection head
```

**Key Parameters:**
- **from**: Input layer index (-1 = previous layer, list for concatenation)
- **number**: Repeat count (scaled by depth_multiple)
- **module**: Module class name
- **args**: Module arguments [channels, kernel, stride, ...]

### 3.2 Example 1: Custom Backbone with Standard Neck

Create file: `ultralytics/cfg/models/custom/my-custom-backbone.yaml`

```yaml
# Ultralytics Custom Model with Custom Backbone
# This example shows a custom backbone using our custom modules

# Parameters
nc: 80  # number of classes
scales:
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # nano
  s: [0.33, 0.50, 1024]  # small
  m: [0.67, 0.75, 768]   # medium
  l: [1.00, 1.00, 512]   # large
  x: [1.00, 1.25, 512]   # xlarge

# Custom Backbone
backbone:
  # [from, repeats, module, args]
  
  # Stem
  - [-1, 1, Conv, [64, 6, 2, 2]]  # 0-P1/2 - Initial conv with stride 2
  
  # Stage 1 - P2/4
  - [-1, 1, CustomDownsample, [128]]  # 1-P2/4 - Custom downsampling
  - [-1, 2, CustomCSPBlock, [128, True]]  # 2 - Custom CSP block
  
  # Stage 2 - P3/8
  - [-1, 1, CustomDownsample, [256]]  # 3-P3/8
  - [-1, 3, CustomCSPBlock, [256, True]]  # 4
  - [-1, 1, CustomAttentionBlock, [256]]  # 5 - Add attention
  
  # Stage 3 - P4/16
  - [-1, 1, CustomDownsample, [512]]  # 6-P4/16
  - [-1, 3, CustomCSPBlock, [512, True]]  # 7
  - [-1, 1, EfficientChannelAttention, [512]]  # 8 - ECA module
  
  # Stage 4 - P5/32
  - [-1, 1, CustomDownsample, [1024]]  # 9-P5/32
  - [-1, 2, CustomCSPBlock, [1024, True]]  # 10
  - [-1, 1, SPPF, [1024, 5]]  # 11 - Spatial Pyramid Pooling

# Standard YOLO Head (FPN + PAN)
head:
  # Neck - Feature Pyramid Network (FPN)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 8], 1, Concat, [1]]  # 13 - Concat P4 from backbone
  - [-1, 2, C2f, [512, False]]  # 14
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 15
  - [[-1, 5], 1, Concat, [1]]  # 16 - Concat P3 from backbone
  - [-1, 2, C2f, [256, False]]  # 17 (P3/8-small)
  
  # Neck - Path Aggregation Network (PAN)
  - [-1, 1, Conv, [256, 3, 2]]  # 18
  - [[-1, 14], 1, Concat, [1]]  # 19 - Concat from FPN
  - [-1, 2, C2f, [512, False]]  # 20 (P4/16-medium)
  
  - [-1, 1, Conv, [512, 3, 2]]  # 21
  - [[-1, 11], 1, Concat, [1]]  # 22 - Concat from backbone
  - [-1, 2, C2f, [1024, True]]  # 23 (P5/32-large)
  
  # Detection Head
  - [[17, 20, 23], 1, Detect, [nc]]  # 24 - Detect(P3, P4, P5)
```

### 3.3 Example 2: Fully Custom Backbone + Custom Neck

Create file: `ultralytics/cfg/models/custom/my-fully-custom.yaml`

```yaml
# Fully Custom Model - Custom Backbone + Custom Neck

nc: 80  # number of classes
scales:
  s: [0.33, 0.50, 1024]

# Custom Backbone
backbone:
  # Stem
  - [-1, 1, Conv, [32, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [64, 3, 2]]  # 1-P2/4
  
  # Custom residual stages
  - [-1, 3, CustomResidualBlock, [128]]  # 2
  - [-1, 1, CustomDownsample, [256]]  # 3-P3/8
  - [-1, 4, CustomResidualBlock, [256]]  # 4
  - [-1, 1, CustomAttentionBlock, [256]]  # 5 - Add attention
  
  - [-1, 1, CustomDownsample, [512]]  # 6-P4/16
  - [-1, 6, CustomResidualBlock, [512]]  # 7
  - [-1, 1, EfficientChannelAttention, [512]]  # 8
  
  - [-1, 1, CustomDownsample, [1024]]  # 9-P5/32
  - [-1, 3, CustomResidualBlock, [1024]]  # 10
  - [-1, 1, SPPF, [1024, 5]]  # 11

# Custom Neck
head:
  # Top-down pathway with custom neck blocks
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 8], 1, Concat, [1]]  # 13
  - [-1, 2, CustomNeckBlock, [512]]  # 14 - Custom neck fusion
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 15
  - [[-1, 5], 1, Concat, [1]]  # 16
  - [-1, 2, CustomNeckBlock, [256]]  # 17 (P3/8)
  
  # Bottom-up pathway with custom blocks
  - [-1, 1, CustomDownsample, [256]]  # 18
  - [[-1, 14], 1, Concat, [1]]  # 19
  - [-1, 2, CustomNeckBlock, [512]]  # 20 (P4/16)
  
  - [-1, 1, CustomDownsample, [512]]  # 21
  - [[-1, 11], 1, Concat, [1]]  # 22
  - [-1, 2, CustomNeckBlock, [1024]]  # 23 (P5/32)
  
  # Detection Head
  - [[17, 20, 23], 1, Detect, [nc]]  # 24
```

### 3.4 Example 3: MobileNet-Inspired Custom Model

```yaml
# Lightweight Custom Model using Inverted Residuals

nc: 80
scales:
  n: [0.33, 0.25, 1024]

backbone:
  # Stem
  - [-1, 1, Conv, [32, 3, 2]]  # 0-P1/2
  
  # MobileNetV2-style blocks
  - [-1, 1, InvertedResidual, [16, 1, 1]]  # 1-P2/4
  - [-1, 2, InvertedResidual, [24, 2, 6]]  # 2
  - [-1, 3, InvertedResidual, [32, 2, 6]]  # 3-P3/8
  - [-1, 4, InvertedResidual, [64, 2, 6]]  # 4-P4/16
  - [-1, 3, InvertedResidual, [96, 1, 6]]  # 5
  - [-1, 3, InvertedResidual, [160, 2, 6]]  # 6-P5/32
  - [-1, 1, InvertedResidual, [320, 1, 6]]  # 7
  - [-1, 1, SPPF, [320, 5]]  # 8

head:
  # Lightweight neck
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 9
  - [[-1, 5], 1, Concat, [1]]  # 10
  - [-1, 2, InvertedResidual, [96, 1, 6]]  # 11
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 3], 1, Concat, [1]]  # 13
  - [-1, 2, InvertedResidual, [32, 1, 6]]  # 14 (P3/8)
  
  - [-1, 1, Conv, [32, 3, 2]]  # 15
  - [[-1, 11], 1, Concat, [1]]  # 16
  - [-1, 2, InvertedResidual, [96, 1, 6]]  # 17 (P4/16)
  
  - [-1, 1, Conv, [96, 3, 2]]  # 18
  - [[-1, 8], 1, Concat, [1]]  # 19
  - [-1, 2, InvertedResidual, [320, 1, 6]]  # 20 (P5/32)
  
  - [[14, 17, 20], 1, Detect, [nc]]  # 21
```

---

## Step 4: Train Your Custom Model

### 4.1 Using Python API

```python
from ultralytics import YOLO

# Load your custom model configuration
model = YOLO("ultralytics/cfg/models/custom/my-custom-backbone.yaml")

# Train the model
results = model.train(
    data="coco8.yaml",  # Dataset configuration
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU device
    
    # Training hyperparameters
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    
    # Model-specific
    optimizer="auto",  # auto, SGD, Adam, AdamW
    amp=True,  # Automatic Mixed Precision
    
    # Saving
    project="custom_models",
    name="my_custom_model_v1",
    save=True,
    save_period=10,  # Save checkpoint every 10 epochs
)

# Validate the model
metrics = model.val()

# Export the model
model.export(format="onnx")
```

### 4.2 Using CLI

```bash
# Train from scratch
yolo detect train \
  model=ultralytics/cfg/models/custom/my-custom-backbone.yaml \
  data=coco8.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=custom_models \
  name=my_custom_model_v1

# Resume training
yolo detect train \
  model=runs/detect/my_custom_model_v1/weights/last.pt \
  resume=True

# Validate
yolo detect val \
  model=runs/detect/my_custom_model_v1/weights/best.pt \
  data=coco8.yaml

# Predict
yolo detect predict \
  model=runs/detect/my_custom_model_v1/weights/best.pt \
  source=path/to/images
```

### 4.3 Custom Dataset Configuration

Create `custom_dataset.yaml`:

```yaml
# Dataset configuration
path: /path/to/dataset  # dataset root directory
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test  # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  # ... add all your classes

# Download script (optional)
download: |
  # Add download/preparation commands here
```

---

## Advanced Customizations

### 5.1 Custom Loss Function

Create file: `ultralytics/utils/custom_loss.py`

```python
import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


class CustomDetectionLoss(v8DetectionLoss):
    """
    Custom detection loss with additional components.
    
    Extends the standard detection loss with custom terms like
    IoU-aware classification loss or custom attention loss.
    """
    
    def __init__(self, model):
        super().__init__(model)
        self.custom_weight = 0.5  # Weight for custom loss component
    
    def __call__(self, preds, batch):
        """
        Calculate custom loss.
        
        Args:
            preds: Model predictions
            batch: Ground truth batch
        
        Returns:
            loss: Total loss
            loss_items: Individual loss components
        """
        # Get standard detection loss
        loss, loss_items = super().__call__(preds, batch)
        
        # Add custom loss component
        custom_loss = self.custom_loss_component(preds, batch)
        loss += self.custom_weight * custom_loss
        
        # Update loss_items for logging
        loss_items = torch.cat((loss_items, custom_loss.unsqueeze(0)))
        
        return loss, loss_items
    
    def custom_loss_component(self, preds, batch):
        """Implement your custom loss logic here."""
        # Example: IoU-aware classification loss
        # This is just a placeholder - implement your actual loss
        return torch.tensor(0.0, device=preds[0].device)


# Register custom loss in your model
# In ultralytics/nn/tasks.py, modify DetectionModel.init_criterion():
def init_criterion(self):
    """Initialize the loss criterion for the DetectionModel."""
    from ultralytics.utils.custom_loss import CustomDetectionLoss
    return CustomDetectionLoss(self)  # Use custom loss instead
```

### 5.2 Custom Training Loop

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    """
    Custom trainer with modified training loop.
    """
    
    def __init__(self, cfg, overrides=None):
        super().__init__(cfg, overrides)
        # Add custom initializations
        self.custom_metric = 0.0
    
    def _do_train(self, world_size=1):
        """
        Override training loop for custom behavior.
        """
        # You can modify the entire training loop here
        # or call super()._do_train() and add custom logic
        return super()._do_train(world_size)
    
    def preprocess_batch(self, batch):
        """
        Custom batch preprocessing.
        """
        batch = super().preprocess_batch(batch)
        # Add custom preprocessing
        # Example: apply custom augmentations, normalization, etc.
        return batch
    
    def optimizer_step(self):
        """
        Custom optimizer step with gradient modification.
        """
        # Example: gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        # Standard optimizer step
        super().optimizer_step()
    
    def plot_training_samples(self, batch, ni):
        """
        Custom visualization of training samples.
        """
        # Implement custom visualization
        super().plot_training_samples(batch, ni)


# Use custom trainer
from ultralytics import YOLO

model = YOLO("my-custom-model.yaml")
model.trainer = CustomTrainer  # Assign custom trainer class
results = model.train(data="coco8.yaml", epochs=100)
```

### 5.3 Custom Data Augmentation

```python
from ultralytics.data.augment import BaseTransform
import cv2
import numpy as np


class CustomAugmentation(BaseTransform):
    """
    Custom augmentation technique.
    """
    
    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability
    
    def apply_image(self, labels):
        """
        Apply custom image augmentation.
        """
        if np.random.random() < self.probability:
            img = labels["img"]
            
            # Example: Custom color jittering
            img = self.color_jitter(img)
            
            # Example: Custom blur
            if np.random.random() < 0.3:
                img = cv2.GaussianBlur(img, (5, 5), 0)
            
            labels["img"] = img
        
        return labels
    
    def color_jitter(self, img):
        """Apply custom color jittering."""
        # Convert to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Random adjustments
        img_hsv[..., 0] += np.random.randint(-10, 10)  # Hue
        img_hsv[..., 1] *= np.random.uniform(0.8, 1.2)  # Saturation
        img_hsv[..., 2] *= np.random.uniform(0.8, 1.2)  # Value
        
        # Clip values
        img_hsv[..., 0] = np.clip(img_hsv[..., 0], 0, 179)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2], 0, 255)
        
        # Convert back to BGR
        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# Use in transforms
from ultralytics.data.dataset import YOLODataset

class CustomYOLODataset(YOLODataset):
    def build_transforms(self, hyp=None):
        """Build transforms with custom augmentation."""
        transforms = super().build_transforms(hyp)
        # Add custom augmentation to the pipeline
        transforms.transforms.insert(0, CustomAugmentation(probability=0.5))
        return transforms
```

### 5.4 Multi-Scale Training

```python
# Enable in training configuration
model = YOLO("my-custom-model.yaml")

results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    multi_scale=True,  # Enable multi-scale training
    
    # Images will be randomly resized between:
    # imgsz * 0.5 and imgsz * 1.5 during training
    # e.g., 320 to 960 pixels for imgsz=640
)
```

---

## Complete Examples

### Example 1: ResNet-Inspired Custom Backbone

```python
# File: ultralytics/nn/modules/custom_resnet.py

import torch
import torch.nn as nn
from .conv import Conv


class ResBottleneck(nn.Module):
    """ResNet bottleneck block."""
    expansion = 4
    
    def __init__(self, c1, c2, s=1, downsample=None):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, s)
        self.cv3 = nn.Sequential(
            nn.Conv2d(c2, c2 * self.expansion, 1, bias=False),
            nn.BatchNorm2d(c2 * self.expansion)
        )
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.cv1(x)
        out = self.cv2(out)
        out = self.cv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.act(out)


class ResStage(nn.Module):
    """ResNet stage with multiple bottleneck blocks."""
    
    def __init__(self, c1, c2, n, s=2):
        super().__init__()
        downsample = nn.Sequential(
            nn.Conv2d(c1, c2 * ResBottleneck.expansion, 1, s, bias=False),
            nn.BatchNorm2d(c2 * ResBottleneck.expansion)
        ) if s != 1 or c1 != c2 * ResBottleneck.expansion else None
        
        layers = [ResBottleneck(c1, c2, s, downsample)]
        c1 = c2 * ResBottleneck.expansion
        
        for _ in range(1, n):
            layers.append(ResBottleneck(c1, c2))
        
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.blocks(x)
```

YAML configuration:

```yaml
# custom-resnet-yolo.yaml

nc: 80
scales:
  s: [0.33, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 7, 2, 3]]  # 0-Stem
  - [-1, 1, nn.MaxPool2d, [3, 2, 1]]  # 1
  - [-1, 1, ResStage, [64, 3, 1]]  # 2-Stage1
  - [-1, 1, ResStage, [128, 4, 2]]  # 3-Stage2 (P3/8)
  - [-1, 1, ResStage, [256, 6, 2]]  # 4-Stage3 (P4/16)
  - [-1, 1, ResStage, [512, 3, 2]]  # 5-Stage4 (P5/32)
  - [-1, 1, SPPF, [2048, 5]]  # 6

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2f, [512, False]]  # 9
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 3], 1, Concat, [1]]
  - [-1, 2, C2f, [256, False]]  # 12 (P3/8)
  
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 2, C2f, [512, False]]  # 15 (P4/16)
  
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C2f, [1024, True]]  # 18 (P5/32)
  
  - [[12, 15, 18], 1, Detect, [nc]]
```

### Example 2: Transformer-Enhanced Custom Model

```python
# File: ultralytics/nn/modules/custom_transformer.py

import torch
import torch.nn as nn
from .transformer import TransformerBlock


class TransformerStage(nn.Module):
    """Transformer stage for vision processing."""
    
    def __init__(self, c1, c2, n=2, num_heads=8):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.transformers = nn.Sequential(*[
            TransformerBlock(c2, num_heads, c2 * 4)
            for _ in range(n)
        ])
        self.norm = nn.LayerNorm(c2)
    
    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        
        # Reshape for transformer: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformers(x)
        x = self.norm(x)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x
```

### Example 3: Complete Training Script

```python
#!/usr/bin/env python3
"""
Complete training script for custom YOLO model.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to model YAML")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="0", help="CUDA device")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--project", type=str, default="runs/train", help="Project name")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained weights")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize model
    LOGGER.info(f"Loading model from {args.model}")
    model = YOLO(args.model)
    
    # Load pretrained weights if specified
    if args.pretrained:
        LOGGER.info(f"Loading pretrained weights from {args.pretrained}")
        model.load(args.pretrained)
    
    # Train
    LOGGER.info("Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        
        # Hyperparameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Options
        amp=True,
        optimizer="auto",
        close_mosaic=10,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    
    # Validate
    LOGGER.info("Validating model...")
    metrics = model.val()
    
    LOGGER.info("Training complete!")
    LOGGER.info(f"Best model saved to: {model.trainer.best}")
    LOGGER.info(f"mAP50: {metrics.box.map50:.4f}")
    LOGGER.info(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
```

Usage:
```bash
python train_custom.py \
  --model ultralytics/cfg/models/custom/my-custom-backbone.yaml \
  --data coco8.yaml \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --name my_custom_experiment
```

---

## Tips and Best Practices

### 1. **Start Simple**
- Begin with a simple custom backbone
- Test with small datasets (coco8, coco128)
- Verify model builds correctly before full training

### 2. **Debugging**
```python
# Test model building
from ultralytics import YOLO
model = YOLO("my-model.yaml")
print(model.model)  # Print architecture

# Test forward pass
import torch
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    y = model.model(x)
    print([yi.shape for yi in y])
```

### 3. **Channel Compatibility**
- Ensure output channels from backbone match expected inputs in neck
- Use the model info to verify layer dimensions:
```python
model.info(detailed=True, verbose=True)
```

### 4. **Memory Optimization**
- Use gradient checkpointing for large models
- Reduce batch size if OOM errors occur
- Enable mixed precision training (amp=True)

### 5. **Performance Tuning**
- Profile your custom modules:
```python
results = model.train(data="coco8.yaml", epochs=1, profile=True)
```
- Monitor training speed and adjust bottlenecks

### 6. **Transfer Learning**
- Start with pretrained backbone when possible
- Freeze backbone initially, then fine-tune:
```python
model = YOLO("my-custom.yaml")
model.train(data="coco8.yaml", epochs=100, freeze=10)  # Freeze first 10 layers
```

---

## Troubleshooting

### Common Issues

1. **Module not found in parse_model()**
   - Ensure module is imported in `nn/tasks.py`
   - Check module is in `base_modules` or `repeat_modules`

2. **Channel mismatch errors**
   - Verify output channels from each layer
   - Use `model.info()` to inspect architecture

3. **YAML parsing errors**
   - Check indentation (use spaces, not tabs)
   - Verify module arguments match `__init__` signature

4. **Training crashes**
   - Reduce batch size
   - Enable AMP: `amp=True`
   - Check dataset format

5. **Poor performance**
   - Verify augmentations are appropriate
   - Check learning rate schedule
   - Increase training epochs
   - Try different optimizers

---

## Summary

You now have a complete understanding of:

1. ✅ Creating custom modules (backbone blocks, neck blocks)
2. ✅ Registering modules in the Ultralytics framework
3. ✅ Writing YAML configurations for custom architectures
4. ✅ Training custom models with standard YOLO trainers
5. ✅ Advanced customizations (loss, trainer, augmentations)
6. ✅ Debugging and optimization techniques

**Next Steps:**
1. Create your first custom module
2. Define a simple YAML configuration
3. Test with a small dataset
4. Iterate and improve your architecture

Happy model building! 🚀
