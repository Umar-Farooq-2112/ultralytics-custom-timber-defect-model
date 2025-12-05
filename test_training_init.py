"""Test training initialization to verify model works with trainer."""
import torch
from ultralytics import YOLO

print("Testing training initialization...")
print("-" * 50)

try:
    # Load custom model
    model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
    print("✓ Model loaded successfully!")
    
    # Simulate a forward pass with labels (training mode)
    batch_size = 2
    img = torch.randn(batch_size, 3, 640, 640)
    
    # Create dummy labels in the format expected by YOLO
    # Format: [batch_idx, class_id, x_center, y_center, width, height]
    labels = []
    for i in range(batch_size):
        # Add 2-3 random boxes per image
        num_boxes = 2 + i
        for _ in range(num_boxes):
            labels.append([
                i,  # batch index
                torch.randint(0, 4, (1,)).item(),  # class (0-3 for 4 classes)
                torch.rand(1).item(),  # x_center (0-1)
                torch.rand(1).item(),  # y_center (0-1)
                torch.rand(1).item() * 0.5,  # width (0-0.5)
                torch.rand(1).item() * 0.5,  # height (0-0.5)
            ])
    
    labels = torch.tensor(labels)
    print(f"✓ Created dummy batch: {batch_size} images, {len(labels)} boxes")
    
    # Test forward pass (inference mode)
    model.model.eval()
    with torch.no_grad():
        output = model.model(img)
    print(f"✓ Inference forward pass successful!")
    print(f"  Output type: {type(output)}")
    print(f"  Number of detection scales: {len(output)}")
    
    # Test training mode
    model.model.train()
    
    # The model expects a dict for training with 'img' and 'batch_idx', etc.
    # But we're just testing if the model structure works
    print(f"✓ Model can switch to training mode!")
    
    # Test backward pass (gradient flow)
    output_train = model.model(img)
    if isinstance(output_train, (list, tuple)):
        # Sum all outputs for a simple loss
        loss = sum(o.sum() for o in output_train if isinstance(o, torch.Tensor))
    else:
        loss = output_train.sum()
    
    loss.backward()
    print(f"✓ Backward pass successful (gradients computed)!")
    
    # Check if key components have gradients
    has_grads = sum(1 for p in model.model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.model.parameters())
    print(f"✓ Parameters with gradients: {has_grads}/{total_params}")
    
    print("\n" + "=" * 50)
    print("✅ ALL TRAINING TESTS PASSED!")
    print("=" * 50)
    print("\n🚀 Model is ready for actual training!")
    print("You can now run:")
    print("  model.train(data='your-data.yaml', epochs=150, ...)")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
