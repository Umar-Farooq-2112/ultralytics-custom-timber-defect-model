"""Debug stride issue."""
import torch
from ultralytics.nn.tasks import DetectionModel

print("Testing stride initialization...")

model = DetectionModel('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml', nc=4, ch=3, verbose=True)

print(f"\nDetectionModel.stride: {model.stride}")
print(f"DetectionModel._is_custom_model: {model._is_custom_model}")

if hasattr(model, '_custom_model') and model._custom_model is not None:
    print(f"Custom model stride: {model._custom_model.stride}")
    print(f"Custom model head.detect.stride: {model._custom_model.head.detect.stride}")

print(f"\nmodel.model[-1] (should be Detect): {type(model.model[-1])}")
print(f"model.model[-1].stride: {model.model[-1].stride}")
