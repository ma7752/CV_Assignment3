"""
Test MovedObjectDETR model with real VIRAT dataset images.
"""
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

from src.model import MovedObjectDETR
from src.dataloader import MovedObjectDataset
from src.config import Config

# Class names for visualization
CLASS_NAMES = {
    0: 'Unknown',
    1: 'Person', 
    2: 'Car',
    3: 'Vehicle',
    4: 'Object',
    5: 'Bike',
    6: 'No-Object'
}

# Colors for each class
CLASS_COLORS = {
    0: 'gray',
    1: 'red',
    2: 'blue', 
    3: 'green',
    4: 'orange',
    5: 'purple',
    6: 'white'
}

def get_transform():
    """Standard DETR preprocessing transforms."""
    return T.Compose([
        T.Resize((800, 800)),  # DETR expects 800x800
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def denormalize(tensor):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize_predictions(img1, img2, targets, outputs, threshold=0.5, save_path=None):
    """
    Visualize model predictions alongside ground truth.
    
    Args:
        img1: First image tensor [3, H, W]
        img2: Second image tensor [3, H, W]  
        targets: Ground truth dict with 'boxes' and 'labels'
        outputs: Model outputs with 'logits' and 'pred_boxes'
        threshold: Confidence threshold for predictions
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Denormalize images for display
    img1_display = denormalize(img1).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    img2_display = denormalize(img2).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    h, w = img1_display.shape[:2]
    
    # Plot Image 1 (reference)
    axes[0].imshow(img1_display)
    axes[0].set_title('Image 1 (Reference)', fontsize=12)
    axes[0].axis('off')
    
    # Plot Image 2 with Ground Truth boxes
    axes[1].imshow(img2_display)
    axes[1].set_title('Image 2 + Ground Truth', fontsize=12)
    axes[1].axis('off')
    
    # Draw ground truth boxes
    if targets is not None and 'boxes' in targets:
        gt_boxes = targets['boxes'].cpu()
        gt_labels = targets['labels'].cpu() if 'labels' in targets else targets.get('class_labels', torch.zeros(len(gt_boxes))).cpu()
        
        for box, label in zip(gt_boxes, gt_labels):
            cx, cy, bw, bh = box.tolist()
            # Convert from center format to corner format
            x = (cx - bw/2) * w
            y = (cy - bh/2) * h
            box_w = bw * w
            box_h = bh * h
            
            label_idx = int(label)
            color = CLASS_COLORS.get(label_idx, 'white')
            
            rect = patches.Rectangle((x, y), box_w, box_h, 
                                     linewidth=2, edgecolor=color, 
                                     facecolor='none', linestyle='--')
            axes[1].add_patch(rect)
            axes[1].text(x, y-5, f'GT: {CLASS_NAMES.get(label_idx, "?")}', 
                        color=color, fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Plot Image 2 with Predictions
    axes[2].imshow(img2_display)
    axes[2].set_title(f'Image 2 + Predictions (thresh={threshold})', fontsize=12)
    axes[2].axis('off')
    
    # Get predictions
    logits = outputs['logits'][0]  # [100, num_classes]
    pred_boxes = outputs['pred_boxes'][0]  # [100, 4]
    
    # Get probabilities and predictions
    probs = logits.softmax(-1)
    scores, labels = probs[..., :-1].max(-1)  # Exclude no-object class
    
    # Filter by threshold
    keep = scores > threshold
    
    pred_count = keep.sum().item()
    print(f"  Predictions above threshold {threshold}: {pred_count}")
    
    for box, score, label in zip(pred_boxes[keep], scores[keep], labels[keep]):
        cx, cy, bw, bh = box.cpu().tolist()
        x = (cx - bw/2) * w
        y = (cy - bh/2) * h
        box_w = bw * w
        box_h = bh * h
        
        label_idx = int(label)
        color = CLASS_COLORS.get(label_idx, 'cyan')
        
        rect = patches.Rectangle((x, y), box_w, box_h,
                                 linewidth=2, edgecolor=color,
                                 facecolor='none')
        axes[2].add_patch(rect)
        axes[2].text(x, y-5, f'{CLASS_NAMES.get(label_idx, "?")}: {score:.2f}',
                    color=color, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    
    plt.show()

def main():
    print("=" * 60)
    print("Testing MovedObjectDETR with Real VIRAT Images")
    print("=" * 60)
    
    # Setup paths
    annotation_dir = Config.ANNOTATION_DIR
    image_base_dir = Config.DATA_BASE_DIR
    
    print(f"\nDataset paths:")
    print(f"  Annotations: {annotation_dir}")
    print(f"  Images: {image_base_dir}")
    
    # Check paths exist
    if not os.path.exists(annotation_dir):
        print(f"ERROR: Annotation dir not found: {annotation_dir}")
        return
    if not os.path.exists(image_base_dir):
        print(f"ERROR: Image base dir not found: {image_base_dir}")
        return
    
    # Create dataset
    transform = get_transform()
    dataset = MovedObjectDataset(
        annotation_dir=annotation_dir,
        image_base_dir=image_base_dir,
        transform=transform
    )
    
    print(f"\nDataset loaded: {len(dataset)} image pairs")
    
    # Create model
    print("\nLoading model...")
    model = MovedObjectDETR(num_classes=6)
    model.eval()  # Set to evaluation mode
    
    # Create output directory for visualizations
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    # Test with a few samples
    num_samples = min(3, len(dataset))
    print(f"\nTesting with {num_samples} samples...")
    
    for idx in range(num_samples):
        print(f"\n--- Sample {idx + 1} ---")
        
        # Load sample
        img1, img2, target = dataset[idx]
        
        print(f"  Image shapes: img1={img1.shape}, img2={img2.shape}")
        print(f"  Ground truth: {len(target['boxes'])} objects")
        for i, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
            print(f"    Object {i+1}: {CLASS_NAMES.get(int(label), '?')} at {box.tolist()}")
        
        # Add batch dimension
        img1_batch = img1.unsqueeze(0)
        img2_batch = img2.unsqueeze(0)
        
        # Run inference (no targets for inference-only mode)
        print("  Running inference...")
        with torch.no_grad():
            outputs = model(img1_batch, img2_batch, targets=None)
        
        print(f"  Output logits shape: {outputs['logits'].shape}")
        print(f"  Output boxes shape: {outputs['pred_boxes'].shape}")
        
        # Get top predictions (even without training, just to see what comes out)
        logits = outputs['logits'][0]
        probs = logits.softmax(-1)
        top_scores, top_labels = probs[..., :-1].max(-1)
        top_k = 5
        top_indices = top_scores.argsort(descending=True)[:top_k]
        
        print(f"  Top {top_k} predictions (untrained model):")
        for i, idx_pred in enumerate(top_indices):
            score = top_scores[idx_pred].item()
            label = top_labels[idx_pred].item()
            box = outputs['pred_boxes'][0][idx_pred].tolist()
            print(f"    {i+1}. {CLASS_NAMES.get(label, '?')}: {score:.4f} at {[f'{x:.3f}' for x in box]}")
        
        # Visualize
        save_path = f'outputs/visualizations/sample_{idx+1}.png'
        visualize_predictions(
            img1, img2, target, outputs, 
            threshold=0.1,  # Low threshold since model is untrained
            save_path=save_path
        )
    
    # Also test with targets to compute loss
    print("\n" + "=" * 60)
    print("Testing with targets (computing loss)...")
    print("=" * 60)
    
    img1, img2, target = dataset[0]
    img1_batch = img1.unsqueeze(0)
    img2_batch = img2.unsqueeze(0)
    
    # Format targets for DETR
    targets = [{
        'boxes': target['boxes'],
        'class_labels': target['labels']  # DETR expects 'class_labels'
    }]
    
    with torch.no_grad():
        outputs = model(img1_batch, img2_batch, targets=targets)
    
    print(f"\nLoss: {outputs.loss.item():.4f}")
    if 'loss_dict' in outputs:
        print("Loss breakdown:")
        for k, v in outputs.loss_dict.items():
            print(f"  {k}: {v.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Real data test COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()