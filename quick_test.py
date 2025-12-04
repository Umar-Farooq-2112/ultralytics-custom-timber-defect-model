"""Quick test of training pipeline."""

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

print("="*80)
print("QUICK TRAINING TEST")
print("="*80)

# Load model
print("\n1. Loading model...")
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
print(f"   ✓ Model loaded: {type(model.model).__name__}")

# Set args
print("\n2. Setting training args...")
model.model.args = get_cfg()
print("   ✓ Args set")

# Create dummy batch
print("\n3. Creating dummy batch...")
batch = {
    'img': torch.randn(4, 3, 640, 640),
    'cls': torch.randint(0, 4, (20, 1)).float(),
    'bboxes': torch.rand(20, 4),
    'batch_idx': torch.cat([torch.zeros(5), torch.ones(5), torch.full((5,), 2), torch.full((5,), 3)]).long(),
}
print(f"   ✓ Batch created: {batch['img'].shape}")

# Forward pass
print("\n4. Forward pass (training mode)...")
model.model.train()
loss, loss_items = model.model(batch)
print(f"   ✓ Loss computed: {loss.sum().item():.4f}")
print(f"   ✓ Box loss: {loss_items[0]:.4f}")
print(f"   ✓ Cls loss: {loss_items[1]:.4f}")
print(f"   ✓ DFL loss: {loss_items[2]:.4f}")

# Backward pass
print("\n5. Backward pass...")
model.model.zero_grad()
loss.sum().backward()
print("   ✓ Gradients computed")

# Check gradients
grad_count = sum(1 for p in model.model.parameters() if p.grad is not None)
param_count = sum(1 for p in model.model.parameters())
print(f"   ✓ {grad_count}/{param_count} parameters have gradients")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✓")
print("="*80)
print("\nYour model is ready for training!")
print("You can now:")
print("1. Push this code to Kaggle/Colab")
print("2. Run: model.train(data='your-dataset.yaml', epochs=100, ...)")
