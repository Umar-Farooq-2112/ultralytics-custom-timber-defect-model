"""Test model loading to verify the fix works."""
import torch
from ultralytics import YOLO

print("Testing custom model loading...")
print("-" * 50)

try:
    # Load custom model
    model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
    print("✓ Model loaded successfully!")
    print(f"✓ Model type: {type(model.model)}")
    print(f"✓ Model class: {model.model.__class__.__name__}")
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model.model(x)
    print(f"✓ Forward pass successful!")
    print(f"✓ Output type: {type(output)}")
    
    # Check if it's the custom model
    inner_model = model.model
    print(f"✓ Inner model type: {type(inner_model).__name__}")
    
    # Try to access custom model properties
    if hasattr(inner_model, 'backbone'):
        print(f"✓ Has backbone: {type(inner_model.backbone).__name__}")
    if hasattr(inner_model, 'neck'):
        print(f"✓ Has neck: {type(inner_model.neck).__name__}")
    if hasattr(inner_model, 'head'):
        print(f"✓ Has head: {type(inner_model.head).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in inner_model.parameters())
    print(f"✓ Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED - Model is ready for training!")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
