"""Quick debug to see what Detect head returns."""

import torch
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

dummy_input = torch.randn(2, 3, 640, 640)

model.model.eval()
with torch.no_grad():
    outputs = model.model(dummy_input)

print(f"Type: {type(outputs)}")
print(f"Length: {len(outputs) if isinstance(outputs, (tuple, list)) else 'N/A'}")

if isinstance(outputs, (tuple, list)):
    for i, out in enumerate(outputs):
        print(f"Output {i}: type={type(out)}, shape={out.shape if hasattr(out, 'shape') else 'N/A'}")
else:
    print(f"Shape: {outputs.shape}")
