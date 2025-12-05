"""Check the model structure after loading."""
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/custom/mobilenetv3-yolo.yaml')

print("Model structure:")
print(f"type(model): {type(model)}")
print(f"type(model.model): {type(model.model)}")
print(f"model.model.__class__.__name__: {model.model.__class__.__name__}")

print("\nAttributes of model.model:")
for attr in dir(model.model):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\nChecking if model.model is DetectionModel:")
from ultralytics.nn.tasks import DetectionModel
print(f"isinstance(model.model, DetectionModel): {isinstance(model.model, DetectionModel)}")

print("\nChecking model.model.model (the actual layers):")
if hasattr(model.model, 'model'):
    print(f"type(model.model.model): {type(model.model.model)}")
    print(f"len(model.model.model): {len(model.model.model)}")
    print("\nmodel.model.model contents:")
    for i, m in enumerate(model.model.model):
        print(f"  [{i}] {type(m).__name__}")
