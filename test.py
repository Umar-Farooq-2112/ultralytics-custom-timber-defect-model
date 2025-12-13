from ultralytics import YOLO
import torch

print("="*80)
print("MobileNetV3-YOLO Integration Test")
print("="*80)

# Load model
print("\n1. Loading model from YAML config...")
model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
print("✓ Model loaded successfully!")

# Check model properties
print(f"\n2. Model Properties:")
print(f"   - Model name: {model.model_name}")
print(f"   - Task: {model.task}")
print(f"   - Model type: {type(model.model).__name__}")
print(f"   - Number of classes: {model.model.nc}")

# Test forward pass
print(f"\n3. Testing forward pass...")
x = torch.randn(1, 3, 320, 320)
with torch.no_grad():
    outputs = model.model(x)

print(f"   ✓ Forward pass successful!")
print(f"   - Number of outputs: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"   - Output {i}: {out.shape}")

# Model info
print(f"\n4. Model Summary:")
model.info(detailed=False, verbose=True)

print("\n" + "="*80)
print("✅ All tests passed! Your model is ready to train!")
print("="*80)
print("\nNext steps:")
print("  python train_custom_model.py")
print("\nOr quick training:")
print("  model.train(data='coco8.yaml', epochs=3, batch=8)")

total_params = sum(p.numel() for p in model.model.parameters())
print(f"\nTotal number of parameters: {total_params}")
number_of_parameters = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {number_of_parameters}")