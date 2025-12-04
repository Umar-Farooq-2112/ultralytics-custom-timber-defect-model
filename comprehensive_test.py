"""
Comprehensive test to validate MobileNetV3-YOLO implementation.
Tests: model loading, forward pass, loss calculation, backward pass, and full training epoch.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
import tempfile
import os
from pathlib import Path


def test_1_model_loading():
    """Test 1: Model loads successfully."""
    print("\n" + "="*80)
    print("TEST 1: Model Loading")
    print("="*80)
    
    model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')
    
    assert model.model is not None, "Model should not be None"
    assert model.task == 'detect', f"Task should be 'detect', got {model.task}"
    
    print(f"✓ Model loaded: {type(model.model).__name__}")
    print(f"✓ Task: {model.task}")
    print(f"✓ Model has {sum(p.numel() for p in model.model.parameters()):,} parameters")
    
    return model


def test_2_forward_inference(model):
    """Test 2: Forward pass with tensor input (inference mode)."""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass (Inference)")
    print("="*80)
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 640, 640)
    
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(dummy_input)
    
    assert isinstance(outputs, (tuple, list)), "Output should be tuple/list"
    
    # Detect head returns (predictions, feature_list)
    # predictions is concatenated, feature_list contains the raw feature maps
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Number of outputs: {len(outputs)}")
    
    if len(outputs) >= 1:
        preds = outputs[0]
        print(f"✓ Predictions shape: {preds.shape}")
    
    if len(outputs) >= 2 and isinstance(outputs[1], list):
        print(f"✓ Number of feature maps: {len(outputs[1])}")
        for i, feat in enumerate(outputs[1]):
            print(f"  - Feature P{i+3} shape: {feat.shape}")
    
    return outputs


def test_3_forward_training_dict(model):
    """Test 3: Forward pass with dict input (training mode)."""
    print("\n" + "="*80)
    print("TEST 3: Forward Pass (Training with dict input)")
    print("="*80)
    
    # Create dummy batch dict like trainer would
    batch = {
        'img': torch.randn(4, 3, 640, 640),
        'cls': torch.randint(0, 4, (20, 1)).float(),
        'bboxes': torch.rand(20, 4),
        'batch_idx': torch.cat([torch.zeros(5), torch.ones(5), torch.full((5,), 2), torch.full((5,), 3)]).long(),
    }
    
    # Set args for loss calculation
    model.model.args = get_cfg()  # Get SimpleNamespace with all hyperparameters
    
    model.model.train()
    try:
        result = model.model(batch)
        
        # Should return (loss, loss_items)
        assert isinstance(result, tuple), f"Training forward should return tuple, got {type(result)}"
        assert len(result) == 2, f"Expected (loss, loss_items), got {len(result)} items"
        
        loss, loss_items = result
        print(f"✓ Forward with dict input successful")
        print(f"✓ Total loss: {loss.sum().item():.4f}")
        print(f"✓ Loss breakdown: box={loss_items[0]:.4f}, cls={loss_items[1]:.4f}, dfl={loss_items[2]:.4f}")
        
        return loss, loss_items
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_4_backward_pass(model):
    """Test 4: Backward pass and gradient computation."""
    print("\n" + "="*80)
    print("TEST 4: Backward Pass")
    print("="*80)
    
    # Create dummy batch
    batch = {
        'img': torch.randn(4, 3, 640, 640),
        'cls': torch.randint(0, 4, (20, 1)).float(),
        'bboxes': torch.rand(20, 4),
        'batch_idx': torch.cat([torch.zeros(5), torch.ones(5), torch.full((5,), 2), torch.full((5,), 3)]).long(),
    }
    
    # Set args
    model.model.args = get_cfg()
    model.model.train()
    
    # Zero gradients
    model.model.zero_grad()
    
    # Forward pass
    loss, loss_items = model.model(batch)
    
    # Backward pass
    loss.sum().backward()  # Sum the loss components before backward
    
    # Check gradients exist
    grad_count = 0
    param_count = 0
    for name, param in model.model.named_parameters():
        param_count += 1
        if param.grad is not None:
            grad_count += 1
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Parameters with gradients: {grad_count}/{param_count}")
    print(f"✓ Backward pass successful")
    
    assert grad_count > 0, "No gradients computed!"


def test_5_optimizer_step(model):
    """Test 5: Full optimization step."""
    print("\n" + "="*80)
    print("TEST 5: Optimization Step")
    print("="*80)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001)
    
    # Create batch
    batch = {
        'img': torch.randn(4, 3, 640, 640),
        'cls': torch.randint(0, 4, (20, 1)).float(),
        'bboxes': torch.rand(20, 4),
        'batch_idx': torch.cat([torch.zeros(5), torch.ones(5), torch.full((5,), 2), torch.full((5,), 3)]).long(),
    }
    
    model.model.args = get_cfg()
    model.model.train()
    
    # Training step
    optimizer.zero_grad()
    loss, loss_items = model.model(batch)
    loss.sum().backward()  # Sum the loss components
    optimizer.step()
    
    print(f"✓ Total loss: {loss.sum().item():.4f}")
    print(f"✓ Optimizer step successful")


def test_6_full_epoch_dummy_data(model):
    """Test 6: Complete training epoch on dummy dataset."""
    print("\n" + "="*80)
    print("TEST 6: Full Training Epoch (Dummy Data)")
    print("="*80)
    
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        # Create dataset structure
        (data_dir / 'images' / 'train').mkdir(parents=True)
        (data_dir / 'labels' / 'train').mkdir(parents=True)
        
        # Create dummy images and labels
        num_samples = 8
        for i in range(num_samples):
            # Create dummy image (640x640 RGB)
            img = torch.randint(0, 255, (640, 640, 3), dtype=torch.uint8)
            img_path = data_dir / 'images' / 'train' / f'img_{i}.jpg'
            
            # Save as numpy array (simpler than PIL)
            import numpy as np
            from PIL import Image
            Image.fromarray(img.numpy()).save(img_path)
            
            # Create dummy label (YOLO format: class x_center y_center width height)
            num_boxes = torch.randint(1, 4, (1,)).item()
            labels = []
            for _ in range(num_boxes):
                cls = torch.randint(0, 4, (1,)).item()
                x, y, w, h = torch.rand(4).tolist()
                labels.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            
            label_path = data_dir / 'labels' / 'train' / f'img_{i}.txt'
            label_path.write_text('\n'.join(labels))
        
        # Create data.yaml
        data_yaml = data_dir / 'data.yaml'
        yaml_content = f"""
path: {data_dir}
train: images/train
val: images/train

nc: 4
names: ['class0', 'class1', 'class2', 'class3']
"""
        data_yaml.write_text(yaml_content)
        
        print(f"✓ Created dummy dataset with {num_samples} samples")
        print(f"✓ Dataset path: {data_dir}")
        
        # Train for 1 epoch
        try:
            print("\nStarting training...")
            results = model.train(
                data=str(data_yaml),
                epochs=1,
                batch=4,
                imgsz=640,
                device='cpu',
                workers=0,  # No multiprocessing for testing
                verbose=True,
                patience=0,  # Disable early stopping
                plots=False,  # Disable plotting
                val=False,  # Skip validation
            )
            
            print("\n✓ Training completed successfully!")
            print(f"✓ Results: {results}")
            
            return results
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Run all tests."""
    print("="*80)
    print("COMPREHENSIVE MOBILENETV3-YOLO TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Model loading
        model = test_1_model_loading()
        
        # Test 2: Inference forward pass
        test_2_forward_inference(model)
        
        # Test 3: Training forward pass with dict
        test_3_forward_training_dict(model)
        
        # Test 4: Backward pass
        test_4_backward_pass(model)
        
        # Test 5: Optimizer step
        test_5_optimizer_step(model)
        
        # Test 6: Full training epoch
        test_6_full_epoch_dummy_data(model)
        
        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nModel is ready for production training on real datasets!")
        print("\nNext steps:")
        print("1. Push your code to the cloud")
        print("2. Run full training with:")
        print("   model.train(data='your-dataset.yaml', epochs=100, ...)")
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
