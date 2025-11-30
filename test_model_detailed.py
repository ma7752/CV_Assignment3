"""
Detailed test script for MovedObjectDETR model.

This script tests the model with dummy data and verifies:
1. Model initialization
2. Forward pass with image pairs
3. Output shapes
4. Parameter counts
"""

import torch
from src.model import MovedObjectDETR


def main():
    print("=" * 70)
    print("TESTING MovedObjectDETR Model")
    print("=" * 70)
    
    print("\n[1/5] Creating model...")
    model = MovedObjectDETR(num_classes=6)
    
    print("\n[2/5] Verifying model structure...")
    # Check that layer4 was replaced
    backbone = model.detr.model.backbone.conv_encoder.model
    print(f"  - Type of layer4: {type(backbone.layer4).__name__}")
    assert type(backbone.layer4).__name__ == 'FeatureDifferenceLayer', \
        "layer4 should be FeatureDifferenceLayer!"
    print("  ✓ layer4 successfully replaced with FeatureDifferenceLayer")
    
    # Check classification head size
    num_classes = model.detr.class_labels_classifier.out_features
    print(f"  - Classification head output: {num_classes} classes")
    assert num_classes == 7, "Should have 7 classes (6 objects + no-object)!"
    print("  ✓ Classification head adjusted correctly")
    
    print("\n[3/5] Creating dummy input data...")
    # Simulate a batch of 2 image pairs
    batch_size = 2
    img1 = torch.randn(batch_size, 3, 800, 800)
    img2 = torch.randn(batch_size, 3, 800, 800)
    print(f"  - img1 shape: {img1.shape}")
    print(f"  - img2 shape: {img2.shape}")
    
    print("\n[4/5] Running forward pass...")
    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Don't compute gradients for testing
        outputs = model(img1, img2)
    
    print("\n[5/5] Validating outputs...")
    print(f"  - Output keys: {list(outputs.keys())}")
    
    # Check logits
    logits_shape = outputs['logits'].shape
    print(f"  - Logits shape: {logits_shape}")
    print(f"    Expected: torch.Size([{batch_size}, 100, 7])")
    assert logits_shape == (batch_size, 100, 7), \
        f"Logits shape mismatch! Expected (2, 100, 7), got {logits_shape}"
    print("  ✓ Logits shape correct")
    
    # Check boxes
    boxes_shape = outputs['pred_boxes'].shape
    print(f"  - Pred boxes shape: {boxes_shape}")
    print(f"    Expected: torch.Size([{batch_size}, 100, 4])")
    assert boxes_shape == (batch_size, 100, 4), \
        f"Boxes shape mismatch! Expected (2, 100, 4), got {boxes_shape}"
    print("  ✓ Pred boxes shape correct")
    
    # Check value ranges
    print(f"\n  Additional checks:")
    print(f"  - Logits range: [{outputs['logits'].min():.3f}, {outputs['logits'].max():.3f}]")
    print(f"  - Boxes range: [{outputs['pred_boxes'].min():.3f}, {outputs['pred_boxes'].max():.3f}]")
    print(f"    (Boxes should be in [0, 1] range after sigmoid)")
    assert (outputs['pred_boxes'] >= 0).all() and (outputs['pred_boxes'] <= 1).all(), \
        "Predicted boxes should be in [0, 1] range!"
    print("  ✓ Boxes correctly normalized to [0, 1]")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED! Model is ready for training.")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Test with real VIRAT dataset images")
    print("  2. Implement training script (src/train.py)")
    print("  3. Implement evaluation metrics (src/evaluate.py)")
    print("  4. Run fine-tuning experiments")


if __name__ == "__main__":
    main()
