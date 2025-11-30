"""Debug DETR structure to find position embedding."""
from transformers import DetrForObjectDetection

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

print("DETR Model Structure:")
print("\nTop level attributes:")
for name, _ in model.named_children():
    print(f"  - {name}")

print("\nDETR.model attributes:")
for name, _ in model.model.named_children():
    print(f"  - model.{name}")

print("\nDETR.model.backbone attributes:")
for name, _ in model.model.backbone.named_children():
    print(f"  - model.backbone.{name}")

print("\nSearching for position_embedding...")
def find_attr(obj, target, prefix=""):
    for name in dir(obj):
        if 'position' in name.lower() and 'embedding' in name.lower():
            print(f"  Found: {prefix}.{name}")

find_attr(model, "position", "model")
find_attr(model.model, "position", "model.model")
if hasattr(model.model, 'backbone'):
    find_attr(model.model.backbone, "position", "model.model.backbone")