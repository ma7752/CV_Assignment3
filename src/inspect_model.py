from transformers import DetrForObjectDetection
import torch

# Load the pre-trained DETR model
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

print("=" * 80)
print("FULL MODEL STRUCTURE")
print("=" * 80)
print(model)

print("\n" + "=" * 80)
print("BACKBONE STRUCTURE (ResNet50)")
print("=" * 80)
print(model.model.backbone)

print("\n" + "=" * 80)
print("BACKBONE LAYERS")
print("=" * 80)
for name, module in model.model.backbone.named_children():
    print(f"{name}: {type(module)}")