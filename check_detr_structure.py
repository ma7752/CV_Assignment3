from transformers import DetrForObjectDetection
import torch

# Load model
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

print("DetrForObjectDetection structure:")
print("=" * 60)
for name, module in model.named_children():
    print(f"- {name}: {type(module).__name__}")

print("\nDetrModel (model.model) structure:")
print("=" * 60)
for name, module in model.model.named_children():
    print(f"- {name}: {type(module).__name__}")

print("\nLet's test the normal forward pass to understand the flow:")
print("=" * 60)

# Create dummy input
dummy_img = torch.randn(1, 3, 800, 800)
pixel_mask = torch.ones(1, 800, 800, dtype=torch.long)

# Get backbone output
backbone_output = model.model.backbone(dummy_img, pixel_mask)
print(f"Backbone output keys: {backbone_output.keys()}")
print(f"Feature maps shape: {backbone_output['feature_maps'][0].shape}")
print(f"Masks shape: {backbone_output['masks'][0].shape if backbone_output['masks'] else 'None'}")

# Check if there's a simpler way
print("\n\nChecking forward signature of DetrModel:")
print("=" * 60)
import inspect
sig = inspect.signature(model.model.forward)
print(f"Parameters: {list(sig.parameters.keys())}")
