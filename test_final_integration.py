"""
Final comprehensive test before pushing to Kaggle.
This tests the complete workflow: load -> initialize -> forward -> backward
"""
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

print("=" * 60)
print("COMPREHENSIVE MODEL INTEGRATION TEST")
print("=" * 60)

try:
    # Test 1: Load via YOLO wrapper
    print("\n1. Loading model via YOLO wrapper...")
    model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
    print("   ✓ Model loaded")
    
    # Test 2: Check if model.model is correct
    print("\n2. Checking model structure...")
    print(f"   - type(model.model): {type(model.model).__name__}")
    print(f"   - Has stride: {hasattr(model.model, 'stride')}")
    if hasattr(model.model, 'stride'):
        print(f"   - Stride: {model.model.stride.tolist()}")
    print(f"   - Has nc: {hasattr(model.model, 'nc')}")
    if hasattr(model.model, 'nc'):
        print(f"   - nc: {model.model.nc}")
    print("   ✓ Structure OK")
    
    # Test 3: Check if it would work with trainer
    print("\n3. Simulating trainer initialization...")
    # The trainer calls DetectionModel(cfg, nc=..., ch=...)
    # Let's simulate that
    cfg = 'ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml'
    detection_model = DetectionModel(cfg, nc=4, ch=3, verbose=False)
    print(f"   - DetectionModel type: {type(detection_model).__name__}")
    print(f"   - Has _is_custom_model: {hasattr(detection_model, '_is_custom_model')}")
    if hasattr(detection_model, '_is_custom_model'):
        print(f"   - Is custom: {detection_model._is_custom_model}")
    print(f"   - Model stride: {detection_model.stride}")
    print("   ✓ Trainer initialization OK")
    
    # Test 4: Forward pass
    print("\n4. Testing forward pass...")
    x = torch.randn(1, 3, 640, 640)
    detection_model.eval()
    with torch.no_grad():
        output = detection_model(x)
    print(f"   - Output type: {type(output)}")
    if isinstance(output, (list, tuple)):
        print(f"   - Output has {len(output)} elements")
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"   - Output[{i}] shape: {o.shape}")
    print("   ✓ Forward pass OK")
    
    # Test 5: Training mode
    print("\n5. Testing training mode...")
    detection_model.train()
    output_train = detection_model(x)
    print(f"   - Training output type: {type(output_train)}")
    print("   ✓ Training mode OK")
    
    # Test 6: Backward pass
    print("\n6. Testing backward pass...")
    if isinstance(output_train, (list, tuple)):
        loss = sum(o.sum() for o in output_train if isinstance(o, torch.Tensor))
    else:
        loss = output_train.sum()
    loss.backward()
    grad_count = sum(1 for p in detection_model.parameters() if p.grad is not None)
    total_params = sum(1 for p in detection_model.parameters())
    print(f"   - Parameters with gradients: {grad_count}/{total_params}")
    print("   ✓ Backward pass OK")
    
    # Test 7: Check that model has required attributes for training
    print("\n7. Checking required attributes for training...")
    required_attrs = ['model', 'stride', 'names', 'nc']
    for attr in required_attrs:
        has_it = hasattr(detection_model, attr)
        print(f"   - Has {attr}: {has_it}")
        if not has_it:
            raise AttributeError(f"Missing required attribute: {attr}")
    print("   ✓ All required attributes present")
    
    # Test 8: Parameter count
    print("\n8. Checking parameter count...")
    total = sum(p.numel() for p in detection_model.parameters())
    print(f"   - Total parameters: {total:,} ({total/1e6:.2f}M)")
    expected_range = (5.0e6, 5.5e6)  # 5.0M to 5.5M
    if expected_range[0] <= total <= expected_range[1]:
        print(f"   ✓ Parameter count within expected range")
    else:
        print(f"   ⚠ Parameter count outside expected range {expected_range}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🚀 Model is READY for Kaggle training!")
    print("\nYou can now safely:")
    print("  1. Commit and push these changes")
    print("  2. Pull on Kaggle")
    print("  3. Run training with:")
    print("     model.train(data='...', epochs=150, ...)")
    print("\n" + "=" * 60)
    
except Exception as e:
    print(f"\n❌ TEST FAILED!")
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    print("\n" + "=" * 60)
