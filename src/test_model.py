"""test_model.py
Test script to verify FeatureDiffExtractor works correctly.

This script performs several tests:
1. Basic shape validation with dummy tensors
2. Test with real images from the dataset
3. Verify gradient flow (for training)
4. Memory profiling (optional)
5. Visualization of feature maps (optional)

Run with:
    python src/test_model.py
"""

import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import FeatureDiffExtractor
from src.config import Config


def test_basic_shapes():
    """Test 1: Verify output shapes with dummy tensors."""
    print("=" * 60)
    print("TEST 1: Basic Shape Validation")
    print("=" * 60)
    
    # Create extractor (pretrained=False for faster testing)
    extractor = FeatureDiffExtractor(pretrained=False)
    extractor.eval()  # Set to eval mode
    
    # Create dummy images of different sizes
    test_sizes = [
        (800, 800),   # Square
        (640, 480),   # 4:3 aspect ratio
        (1024, 768),  # Larger image
    ]
    
    for h, w in test_sizes:
        print(f"\nTesting with input size: {h}x{w}")
        
        # Create random tensors (batch_size=2)
        img1 = torch.randn(2, 3, h, w)
        img2 = torch.randn(2, 3, h, w)
        
        # Forward pass
        with torch.no_grad():
            proj_feat, mask = extractor(img1, img2)
        
        # Expected spatial dimensions after ResNet layer3
        # ResNet downsamples by factor of 16 after layer3
        expected_h = h // 16
        expected_w = w // 16
        
        print(f"  Input shape:    {img1.shape}")
        print(f"  proj_feat shape: {proj_feat.shape}")
        print(f"  mask shape:     {mask.shape}")
        print(f"  Expected H', W': {expected_h}, {expected_w}")
        
        # Assertions
        assert proj_feat.shape == (2, 256, expected_h, expected_w), \
            f"Expected shape (2, 256, {expected_h}, {expected_w}), got {proj_feat.shape}"
        assert mask.shape == (2, expected_h, expected_w), \
            f"Expected mask shape (2, {expected_h}, {expected_w}), got {mask.shape}"
        assert mask.dtype == torch.bool, f"Mask should be bool, got {mask.dtype}"
        assert not mask.any(), "Mask should be all False (no padding)"
        
        print("  ✓ Shapes are correct!")
    
    print("\n✅ TEST 1 PASSED: All shapes are correct\n")


def test_with_real_data():
    """Test 2: Load real images from dataset and verify processing."""
    print("=" * 60)
    print("TEST 2: Real Dataset Images")
    print("=" * 60)
    
    # Use paths from config
    data_base_dir = Config.DATA_BASE_DIR
    annotation_dir = Config.ANNOTATION_DIR
    
    print(f"\nData base dir: {data_base_dir}")
    print(f"Annotation dir: {annotation_dir}")
    
    # Check if directories exist
    if not os.path.exists(data_base_dir):
        print(f"⚠️  WARNING: {data_base_dir} not found, skipping real data test")
        print("\n✅ TEST 2 SKIPPED: Data directory not found\n")
        return
    
    if not os.path.exists(annotation_dir):
        print(f"⚠️  WARNING: {annotation_dir} not found, skipping real data test")
        print("\n✅ TEST 2 SKIPPED: Annotation directory not found\n")
        return
    
    # Get list of pair directories
    pair_dirs = [d for d in os.listdir(data_base_dir) 
                 if os.path.isdir(os.path.join(data_base_dir, d)) and d.startswith('Pair_')]
    
    if not pair_dirs:
        print("⚠️  WARNING: No Pair_* directories found")
        print("\n✅ TEST 2 SKIPPED: No data found\n")
        return
    
    print(f"\nFound {len(pair_dirs)} pair directories")
    print(f"Testing with first pair: {pair_dirs[0]}")
    
    # Use first pair directory
    test_pair_dir = os.path.join(data_base_dir, pair_dirs[0])
    
    # Get image files in this directory
    image_files = [f for f in os.listdir(test_pair_dir) if f.endswith('.png')]
    
    if len(image_files) < 2:
        print(f"⚠️  WARNING: Need at least 2 images, found {len(image_files)}")
        print("\n✅ TEST 2 SKIPPED: Insufficient images\n")
        return
    
    print(f"Found images: {image_files}")
    
    # Create transforms
    transform = T.Compose([
        T.Resize((Config.DETR_INPUT_SIZE, Config.DETR_INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the two images
    img1_path = os.path.join(test_pair_dir, image_files[0])
    img2_path = os.path.join(test_pair_dir, image_files[1])
    
    print(f"\nLoading images:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    
    try:
        img1_pil = Image.open(img1_path).convert('RGB')
        img2_pil = Image.open(img2_path).convert('RGB')
        
        print(f"  Original sizes: {img1_pil.size}, {img2_pil.size}")
        
        # Apply transforms
        img1 = transform(img1_pil)
        img2 = transform(img2_pil)
        
        print(f"  Transformed shapes: {img1.shape}, {img2.shape}")
        
    except Exception as e:
        print(f"⚠️  ERROR loading images: {e}")
        print("\n✅ TEST 2 SKIPPED: Image loading failed\n")
        return
    
    # Create extractor
    print("\nCreating feature extractor (with pretrained weights)...")
    extractor = FeatureDiffExtractor(pretrained=True)
    extractor.eval()
    
    # Add batch dimension
    img1_batch = img1.unsqueeze(0)
    img2_batch = img2.unsqueeze(0)
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        proj_feat, mask = extractor(img1_batch, img2_batch)
    
    print(f"  Input shapes: {img1_batch.shape}, {img2_batch.shape}")
    print(f"  Output proj_feat shape: {proj_feat.shape}")
    print(f"  Output mask shape: {mask.shape}")
    
    # Verify shapes
    assert proj_feat.shape[0] == 1, "Batch size should be 1"
    assert proj_feat.shape[1] == 256, "Should have 256 channels"
    assert mask.shape[0] == 1, "Batch size should be 1"
    
    print("  ✓ Single image pair processing works!")
    
    # Test batch processing with multiple pairs
    print("\nTesting batch processing...")
    batch_img1 = []
    batch_img2 = []
    
    num_test_pairs = min(4, len(pair_dirs))
    
    for pair_dir_name in pair_dirs[:num_test_pairs]:
        pair_path = os.path.join(data_base_dir, pair_dir_name)
        imgs = [f for f in os.listdir(pair_path) if f.endswith('.png')]
        
        if len(imgs) >= 2:
            try:
                img1_pil = Image.open(os.path.join(pair_path, imgs[0])).convert('RGB')
                img2_pil = Image.open(os.path.join(pair_path, imgs[1])).convert('RGB')
                
                batch_img1.append(transform(img1_pil))
                batch_img2.append(transform(img2_pil))
            except Exception as e:
                print(f"  Warning: Could not load pair {pair_dir_name}: {e}")
                continue
    
    if len(batch_img1) > 0:
        batch_img1 = torch.stack(batch_img1)
        batch_img2 = torch.stack(batch_img2)
        
        print(f"  Processing batch of {len(batch_img1)} pairs...")
        
        with torch.no_grad():
            proj_feat, mask = extractor(batch_img1, batch_img2)
        
        print(f"  Batch input shape: {batch_img1.shape}")
        print(f"  Batch output shape: {proj_feat.shape}")
        
        assert proj_feat.shape[0] == len(batch_img1), "Batch size mismatch"
        
        print("  ✓ Batch processing works!")
    
    print("\n✅ TEST 2 PASSED: Real data processing successful\n")


def test_gradient_flow():
    """Test 3: Verify gradients flow correctly for training."""
    print("=" * 60)
    print("TEST 3: Gradient Flow Verification")
    print("=" * 60)
    
    # Create extractor in training mode
    extractor = FeatureDiffExtractor(pretrained=False)
    extractor.train()
    
    # Create dummy inputs that require gradients
    img1 = torch.randn(2, 3, 640, 640, requires_grad=True)
    img2 = torch.randn(2, 3, 640, 640, requires_grad=True)
    
    # Forward pass
    proj_feat, mask = extractor(img1, img2)
    
    # Create dummy loss (sum of features)
    loss = proj_feat.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    print("\nChecking gradients...")
    
    # Check input gradients
    assert img1.grad is not None, "img1 should have gradients"
    assert img2.grad is not None, "img2 should have gradients"
    print("  ✓ Input gradients exist")
    
    # Check model parameter gradients
    has_grad = False
    for name, param in extractor.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"  ✓ {name}: grad shape {param.grad.shape}")
    
    assert has_grad, "At least some parameters should have gradients"
    
    # Specifically check projection layer
    assert extractor.proj.weight.grad is not None, "Projection layer should have gradients"
    print("  ✓ Projection layer has gradients")
    
    print("\n✅ TEST 3 PASSED: Gradients flow correctly\n")


def test_freeze_backbone():
    """Test 4: Verify we can freeze the backbone for fine-tuning."""
    print("=" * 60)
    print("TEST 4: Backbone Freezing")
    print("=" * 60)
    
    extractor = FeatureDiffExtractor(pretrained=False)
    
    # Count trainable parameters before freezing
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_before = count_trainable_params(extractor)
    backbone_params = count_trainable_params(extractor.backbone)
    proj_params = count_trainable_params(extractor.proj)
    
    print(f"\nBefore freezing:")
    print(f"  Total trainable params: {total_before:,}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Projection params: {proj_params:,}")
    
    # Freeze backbone
    for param in extractor.backbone.parameters():
        param.requires_grad = False
    
    total_after = count_trainable_params(extractor)
    
    print(f"\nAfter freezing backbone:")
    print(f"  Total trainable params: {total_after:,}")
    print(f"  Expected (projection only): {proj_params:,}")
    
    assert total_after == proj_params, "Only projection params should be trainable"
    
    # Verify gradients flow only to projection
    extractor.train()
    img1 = torch.randn(1, 3, 640, 640)
    img2 = torch.randn(1, 3, 640, 640)
    
    proj_feat, _ = extractor(img1, img2)
    loss = proj_feat.sum()
    loss.backward()
    
    # Check backbone has no gradients
    for name, param in extractor.backbone.named_parameters():
        assert param.grad is None, f"Frozen backbone param {name} should have no gradient"
    
    # Check projection has gradients
    assert extractor.proj.weight.grad is not None, "Projection should have gradients"
    
    print("  ✓ Backbone successfully frozen")
    print("  ✓ Gradients only flow to projection layer")
    
    print("\n✅ TEST 4 PASSED: Freezing works correctly\n")


def test_memory_usage():
    """Test 5: Profile memory usage (optional)."""
    print("=" * 60)
    print("TEST 5: Memory Profiling (Optional)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU memory test")
        print("   (CPU memory profiling less informative)")
        print("\n✅ TEST 5 SKIPPED: No GPU available\n")
        return
    
    device = torch.device('cuda')
    extractor = FeatureDiffExtractor(pretrained=False).to(device)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure memory for different batch sizes
    batch_sizes = [1, 2, 4, 8]
    
    print("\nMemory usage by batch size:")
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        img1 = torch.randn(bs, 3, 800, 800, device=device)
        img2 = torch.randn(bs, 3, 800, 800, device=device)
        
        with torch.no_grad():
            proj_feat, mask = extractor(img1, img2)
        
        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"  Batch size {bs}: {mem_allocated:.1f} MB allocated, {mem_peak:.1f} MB peak")
    
    print("\n✅ TEST 5 PASSED: Memory profiling complete\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FEATURE DIFF EXTRACTOR TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_basic_shapes()
        test_with_real_data()
        test_gradient_flow()
        test_freeze_backbone()
        test_memory_usage()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour FeatureDiffExtractor is working correctly and ready to")
        print("be integrated with DETR's transformer in the next step.")
        print("\nNext steps:")
        print("  1. Implement DetrWithFeatureDiff wrapper")
        print("  2. Create training script")
        print("  3. Run fine-tuning experiments")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())