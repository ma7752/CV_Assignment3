"""Quick test of DetrWithFeatureDiff wrapper."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from src.model import FeatureDiffExtractor, DetrWithFeatureDiff
from src.config import Config

print("Creating feature extractor...")
extractor = FeatureDiffExtractor(pretrained=False)  # False for faster testing

print("\nCreating complete model...")
model = DetrWithFeatureDiff(
    extractor=extractor,
    num_classes=Config.NUM_CLASSES,
    freeze_backbone=True,
    freeze_transformer=False
)

print("\nTesting forward pass...")
img1 = torch.randn(2, 3, 800, 800)
img2 = torch.randn(2, 3, 800, 800)

# Test inference
model.eval()
outputs = model(img1, img2)
print(f"Logits shape: {outputs['logits'].shape}")
print(f"Boxes shape: {outputs['pred_boxes'].shape}")

# Test with targets
targets = [
    {'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3]]), 'class_labels': torch.tensor([1])},
    {'boxes': torch.tensor([[0.3, 0.4, 0.15, 0.2]]), 'class_labels': torch.tensor([2])}
]

model.train()
outputs = model(img1, img2, targets=targets)
print(f"\nWith targets:")
print(f"  Total loss: {outputs['loss'].item():.4f}")
print(f"  CE loss: {outputs['loss_ce'].item():.4f}")

print("\n✅ Wrapper works correctly!")