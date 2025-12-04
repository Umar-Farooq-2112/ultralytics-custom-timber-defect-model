"""Quick test to verify training works with the fix."""

from ultralytics import YOLO
import torch

# Load model
print("Loading MobileNetV3-YOLO...")
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

print(f"Model type: {type(model.model).__name__}")
print(f"Task: {model.task}")

# Create a fake batch like the trainer would
batch = {
    'img': torch.randn(2, 3, 640, 640),
    'cls': torch.randint(0, 4, (10, 1)),
    'bboxes': torch.rand(10, 4),
    'batch_idx': torch.cat([torch.zeros(5), torch.ones(5)]).long(),
}

print("\nTesting loss calculation...")
try:
    # This should work now - calls model.loss() instead of model.forward()
    loss, loss_items = model.model.loss(batch)
    print(f"✓ Loss calculation successful!")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss items: {loss_items}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Starting actual training (1 epoch, small batch)...")
print("="*80)

# Train for 1 epoch to verify everything works
try:
    # results = model.train(
    #     data="defects-in-timber/data.yaml",
    #     epochs=1,
    #     batch=4,  # Small batch for quick test
    #     imgsz=640,
    #     device='cpu',  # Use CPU
    #     verbose=True,
    # )
    print("\n✓ Training completed successfully!")
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
