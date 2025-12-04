# Common Errors and Quick Fixes

## Simple Solutions to Common Problems

This guide shows the **most common errors** you'll encounter and **exactly how to fix them** with simple explanations.

---

## Table of Contents

1. [Shape/Dimension Errors](#shapedimension-errors)
2. [Import/Module Errors](#importmodule-errors)
3. [Training Errors](#training-errors)
4. [Memory Errors](#memory-errors)
5. [Gradient/Backward Errors](#gradientbackward-errors)
6. [General Tips](#general-tips)

---

## Shape/Dimension Errors

### ❌ Error 1: Size Mismatch in Forward Pass

**Error Message:**
```
RuntimeError: size mismatch, got input (4, 64, 80, 80), expected (4, 32, 80, 80)
```

**What it means:**
- You changed output channels somewhere but didn't update the next layer's input channels
- Layer expecting 32 channels but receiving 64

**How to fix:**

```python
# FIND where you changed output:
self.conv1 = DWConvCustom(24, 64, k=3, s=1)  # Changed 24→64 output

# UPDATE the next layer's input:
# WRONG:
self.conv2 = DWConvCustom(32, 32, k=3, s=1)  # Still expects 32

# CORRECT:
self.conv2 = DWConvCustom(64, 32, k=3, s=1)  # Now expects 64 ✓
```

**Rule:** Output of layer N must equal input of layer N+1

---

### ❌ Error 2: Neck Input Channel Mismatch

**Error Message:**
```
RuntimeError: Given groups=1, expected input[16, 40, 20, 20] to have 24 channels, but got 40 channels
```

**What it means:**
- Backbone output channels don't match neck input channels
- Backbone is outputting [24, 40, 160] but neck expects different

**How to fix:**

```python
# In custom_models.py, UPDATE neck initialization:

# WRONG:
self.backbone = MobileNetV3BackboneDW()  # Outputs [24, 40, 160]
self.neck = UltraLiteNeckDW(in_channels=[32, 64, 128])  # Expects different ✗

# CORRECT:
self.backbone = MobileNetV3BackboneDW()  # Outputs [24, 40, 160]
self.neck = UltraLiteNeckDW(in_channels=[24, 40, 160])  # Match! ✓
```

**Quick check:**
```python
# Test backbone output
x = torch.randn(1, 3, 640, 640)
p3, p4, p5 = backbone(x)
print(p3.shape[1], p4.shape[1], p5.shape[1])  # Check channel counts
# Update neck: in_channels=[p3_ch, p4_ch, p5_ch]
```

---

### ❌ Error 3: Head Input Channel Mismatch

**Error Message:**
```
RuntimeError: shape '[2, 64, -1]' is invalid for input of size 147456
```

**What it means:**
- Neck output channels don't match head input channels

**How to fix:**

```python
# In custom_models.py:

# Step 1: Check neck output channels
neck_out_channels = [32, 48, 64]  # Your neck outputs these

# Step 2: Make sure head uses same channels
# WRONG:
self.head = Detect(nc=nc, ch=(64, 96, 128))  # Different channels ✗

# CORRECT:
self.head = Detect(nc=nc, ch=tuple(neck_out_channels))  # Use same ✓
# OR explicitly:
self.head = Detect(nc=nc, ch=(32, 48, 64))  # Match exactly ✓
```

---

### ❌ Error 4: Interpolation Size Mismatch

**Error Message:**
```
RuntimeError: expected scalar type Float but found Half
# OR
RuntimeError: sizes of tensors must match except in dimension 1
```

**What it means:**
- Trying to add/concatenate features of different spatial sizes
- Common when doing feature fusion

**How to fix:**

```python
# WRONG - Adding features of different sizes:
p4_fused = p4 + p5  # p4 is 40x40, p5 is 20x20 ✗

# CORRECT - Upsample first:
import torch.nn.functional as F
p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
p4_fused = p4 + p5_up  # Now same size ✓

# OR downsample:
p4_down = F.max_pool2d(p4, kernel_size=2)
p4_fused = p4_down + p5  # Now same size ✓
```

---

## Import/Module Errors

### ❌ Error 5: Module Not Found

**Error Message:**
```
ImportError: cannot import name 'YourCustomBlock' from 'ultralytics.nn.modules'
```

**What it means:**
- You created a new module but didn't export it

**How to fix:**

```python
# Step 1: Check module exists in custom_mobilenet_blocks.py
class YourCustomBlock(nn.Module):  # ✓ Defined
    # ...

# Step 2: ADD to modules/__init__.py
__all__ = (
    # ... existing modules ...
    "YourCustomBlock",  # ← ADD THIS (don't forget comma!)
)
```

---

### ❌ Error 6: Circular Import

**Error Message:**
```
ImportError: cannot import name 'X' from partially initialized module 'Y'
```

**What it means:**
- File A imports from File B, File B imports from File A
- Creates a loop

**How to fix:**

```python
# WRONG - Circular import:
# In custom_models.py:
from custom_mobilenet_blocks import Backbone
# In custom_mobilenet_blocks.py:
from custom_models import SomeFunction  # ✗ Creates circle

# CORRECT - Import inside function:
# In custom_mobilenet_blocks.py:
def some_method(self):
    from custom_models import SomeFunction  # ✓ Import locally
    SomeFunction()
```

---

### ❌ Error 7: Attribute Not Found

**Error Message:**
```
AttributeError: 'MobileNetV3YOLO' object has no attribute 'model'
```

**What it means:**
- Loss function expects `model.model` but your class doesn't have it

**How to fix:**

```python
# In custom_models.py __init__:

# ADD this line:
self.model = nn.ModuleList([self.backbone, self.neck, self.head])  # ✓

# The loss function needs model.model[-1] to access the head
```

---

## Training Errors

### ❌ Error 8: Dict Input Error (conv2d received dict)

**Error Message:**
```
TypeError: conv2d() received an invalid combination of arguments - got (dict, ...)
```

**What it means:**
- Training passes a batch dict to forward(), but forward() doesn't handle it

**How to fix:**

```python
# In custom_models.py, UPDATE forward method:

# WRONG:
def forward(self, x):
    feats = self.backbone(x)  # ✗ Fails when x is dict
    # ...

# CORRECT:
def forward(self, x, *args, **kwargs):
    # Check if input is dict (training mode)
    if isinstance(x, dict):
        return self.loss(x, *args, **kwargs)  # ✓ Call loss instead
    
    # Inference mode
    feats = self.backbone(x)
    # ...
```

---

### ❌ Error 9: No Attribute 'args'

**Error Message:**
```
AttributeError: 'MobileNetV3YOLO' object has no attribute 'args'
```

**What it means:**
- Loss function needs `model.args` for hyperparameters

**How to fix:**

```python
# In custom_models.py __init__:

# ADD these lines:
self.args = None  # Will be set by trainer ✓
self.task = 'detect'  # Task type ✓
```

Already fixed in our code, but if you create new model, add these!

---

### ❌ Error 10: Loss is NaN or Inf

**Error Message:**
```
RuntimeError: Function 'MulBackward0' returned nan values in its 0th output
# OR loss shows: nan
```

**What it means:**
- Loss calculation produced invalid values
- Usually from learning rate too high or bad initialization

**How to fix:**

```python
# Option 1: Lower learning rate
model.train(
    lr0=0.0001,  # Try smaller (was 0.01)
    # ...
)

# Option 2: Check for division by zero
# In your custom module:
# WRONG:
output = x / x.sum()  # ✗ Could be zero

# CORRECT:
output = x / (x.sum() + 1e-6)  # ✓ Add epsilon

# Option 3: Clip gradients
model.train(
    # ... other args ...
)
# Add after backward in custom training loop:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

---

### ❌ Error 11: Dataset Not Found

**Error Message:**
```
FileNotFoundError: 'your-dataset.yaml' does not exist
```

**What it means:**
- Path to dataset YAML is wrong

**How to fix:**

```python
# Check paths:
# WRONG:
model.train(data='dataset.yaml')  # ✗ Not in current dir

# CORRECT - Use absolute path:
model.train(data='/full/path/to/dataset.yaml')  # ✓

# OR relative from script location:
model.train(data='./datasets/dataset.yaml')  # ✓

# OR use built-in datasets:
model.train(data='coco8.yaml')  # ✓ Built-in
```

---

## Memory Errors

### ❌ Error 12: CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**What it means:**
- Model + batch too big for GPU memory

**How to fix (try in order):**

```python
# Fix 1: Reduce batch size
model.train(
    batch=8,  # Try 16 → 8 → 4 → 2
    # ...
)

# Fix 2: Reduce image size
model.train(
    batch=16,
    imgsz=512,  # Try 640 → 512 → 416
    # ...
)

# Fix 3: Use CPU (slower but works)
model.train(
    device='cpu',
    workers=0,
    # ...
)

# Fix 4: Enable gradient checkpointing (if your model supports it)
# In custom module:
from torch.utils.checkpoint import checkpoint
output = checkpoint(self.large_module, x)  # Saves memory
```

---

### ❌ Error 13: CPU Memory Error

**Error Message:**
```
MemoryError: Unable to allocate array
# OR system freezes
```

**What it means:**
- Too much data in RAM

**How to fix:**

```python
# Fix 1: Disable caching
model.train(
    cache=False,  # Don't load all images to RAM
    # ...
)

# Fix 2: Reduce workers
model.train(
    workers=0,  # Single process
    # ...
)

# Fix 3: Use smaller dataset fraction
model.train(
    fraction=0.5,  # Use only 50% of data
    # ...
)
```

---

## Gradient/Backward Errors

### ❌ Error 14: No Gradients Computed

**Error Message:**
```
# No error, but training doesn't improve
# OR all gradients are None
```

**What it means:**
- Backward pass not computing gradients

**How to fix:**

```python
# Check 1: Model in train mode
model.model.train()  # Not eval() ✓

# Check 2: No detach() calls
# WRONG:
x = self.layer(x).detach()  # ✗ Stops gradients

# CORRECT:
x = self.layer(x)  # ✓ Keeps gradients

# Check 3: Loss requires_grad
loss, loss_items = model.model(batch)
print(loss.requires_grad)  # Should be True

# If False, check model parameters:
for p in model.model.parameters():
    p.requires_grad = True  # Enable gradients
```

---

### ❌ Error 15: Gradient Modified Error

**Error Message:**
```
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
```

**What it means:**
- You modified a tensor in-place that's needed for gradients

**How to fix:**

```python
# WRONG - In-place operations:
x += self.conv(x)  # ✗ In-place
x *= 2  # ✗ In-place
x[0] = 0  # ✗ In-place

# CORRECT - Create new tensor:
x = x + self.conv(x)  # ✓ New tensor
x = x * 2  # ✓ New tensor
x = torch.cat([torch.zeros(1), x[1:]], dim=0)  # ✓ New tensor

# OR use .clone():
x_copy = x.clone()
x_copy += self.conv(x)  # ✓ Modify copy
```

---

## General Tips

### ✅ Quick Debugging Checklist

When something doesn't work:

1. **Check shapes:**
   ```python
   print(f"Shape: {tensor.shape}")
   ```

2. **Check channels match:**
   ```python
   print(f"Out: {layer1.out_channels}, In: {layer2.in_channels}")
   ```

3. **Check for NaN:**
   ```python
   print(f"Has NaN: {torch.isnan(tensor).any()}")
   ```

4. **Check gradients:**
   ```python
   print(f"Requires grad: {tensor.requires_grad}")
   ```

5. **Test small batch:**
   ```python
   x = torch.randn(1, 3, 640, 640)  # Batch size = 1
   output = model(x)
   ```

---

### ✅ Prevention Tips

**Before making changes:**
- ✓ Save working version (`git commit`)
- ✓ Test on small data first
- ✓ Change one thing at a time

**After making changes:**
- ✓ Run dimension test
- ✓ Run loss test
- ✓ Check model.info()

**During development:**
- ✓ Add print statements to check shapes
- ✓ Use small models for testing
- ✓ Test with batch_size=1 first

---

### ✅ Emergency Quick Fixes

**Model won't load?**
```python
# Start fresh
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
```

**Training crashes?**
```python
# Minimal safe settings
model.train(
    data='your-data.yaml',
    epochs=1,
    batch=2,
    imgsz=320,
    device='cpu',
    workers=0,
)
```

**Shapes wrong?**
```python
# Print everything
def forward(self, x):
    print(f"Input: {x.shape}")
    x = self.layer1(x)
    print(f"After layer1: {x.shape}")
    x = self.layer2(x)
    print(f"After layer2: {x.shape}")
    return x
```

---

## Error Lookup Table

| Error Contains | Most Likely Cause | Quick Fix |
|----------------|-------------------|-----------|
| `size mismatch` | Channel mismatch | Update channel numbers |
| `cannot import` | Missing export | Add to `__init__.py` |
| `received dict` | Wrong forward() | Add `isinstance(x, dict)` check |
| `no attribute` | Missing attribute | Add to `__init__()` |
| `out of memory` | Batch too large | Reduce `batch` or `imgsz` |
| `NaN` or `Inf` | Bad gradients | Lower `lr0` |
| `not found` | Wrong path | Check file paths |
| `modified by inplace` | Using `+=` etc | Use `x = x +` instead |

---

## Still Stuck?

### Debug Script

```python
# debug_model.py
import torch
from ultralytics import YOLO

print("1. Loading model...")
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

print("2. Testing forward pass...")
x = torch.randn(1, 3, 640, 640)
model.model.eval()
with torch.no_grad():
    out = model.model(x)
print(f"   Output: {out[0].shape}")

print("3. Checking parameters...")
total = sum(p.numel() for p in model.model.parameters())
print(f"   Total params: {total:,}")

print("4. Testing training mode...")
from ultralytics.cfg import get_cfg
model.model.args = get_cfg()
model.model.train()

batch = {
    'img': torch.randn(2, 3, 640, 640),
    'cls': torch.randint(0, 4, (10, 1)).float(),
    'bboxes': torch.rand(10, 4),
    'batch_idx': torch.cat([torch.zeros(5), torch.ones(5)]).long(),
}

try:
    loss, loss_items = model.model(batch)
    print(f"   Loss: {loss.sum().item():.4f}")
    print("✓ Everything works!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
```

Run this to find exactly where the problem is!

---

**Remember:** Most errors are just channel mismatches. Check your channels first! 🔍
