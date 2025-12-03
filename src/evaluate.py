'''
Load trained model
Run inference on test set
Calculate precision, recall, and other metrics
Generate visualizations (predicted vs ground truth bounding boxes)
Save results
'''

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict

from config import Config
from model import MovedObjectDETR, convert_targets_to_detr_format
from dataloader import MovedObjectDataset


# =============================================================================
# SECTION 1: IoU and Box Utilities
# =============================================================================

def box_cxcywh_to_xyxy(boxes):
    """
    Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    
    Args:
        boxes: Tensor of shape [N, 4] in (cx, cy, w, h) format, normalized [0, 1]
    
    Returns:
        Tensor of shape [N, 4] in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in (x1, y1, x2, y2) format.
    
    Args:
        box1: Tensor of shape [4]
        box2: Tensor of shape [4]
    
    Returns:
        IoU value (float)
    """
    # Get intersection coordinates
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())
    
    # Compute intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    intersection = inter_width * inter_height
    
    # Compute union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union.item()


def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4] in (x1, y1, x2, y2) format
        boxes2: Tensor of shape [M, 4] in (x1, y1, x2, y2) format
    
    Returns:
        IoU matrix of shape [N, M]
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    iou_matrix = torch.zeros(N, M)
    
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])
    
    return iou_matrix


# =============================================================================
# SECTION 2: Matching Predictions to Ground Truth
# =============================================================================

def match_predictions_to_gt(pred_boxes, pred_labels, pred_scores, 
                             gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes using Hungarian-style greedy matching.
    
    Args:
        pred_boxes: [N, 4] predicted boxes in (x1, y1, x2, y2) format
        pred_labels: [N] predicted class labels
        pred_scores: [N] confidence scores
        gt_boxes: [M, 4] ground truth boxes in (x1, y1, x2, y2) format
        gt_labels: [M] ground truth class labels
        iou_threshold: minimum IoU to consider a match
    
    Returns:
        matches: List of (pred_idx, gt_idx, iou) tuples for true positives
        unmatched_preds: List of pred_idx for false positives
        unmatched_gts: List of gt_idx for false negatives
    """
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []
    
    # Compute IoU matrix
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    
    matches = []
    matched_gt = set()
    matched_pred = set()
    
    # Sort predictions by confidence score (descending)
    sorted_pred_indices = torch.argsort(pred_scores, descending=True)
    
    for pred_idx in sorted_pred_indices.tolist():
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            
            # Check if classes match
            if pred_labels[pred_idx] != gt_labels[gt_idx]:
                continue
            
            iou = iou_matrix[pred_idx, gt_idx].item()
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
    
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    
    return matches, unmatched_preds, unmatched_gts


# =============================================================================
# SECTION 3: Precision, Recall, and AP Calculation
# =============================================================================

def compute_precision_recall(all_predictions, all_ground_truths, 
                              iou_threshold=0.5, score_threshold=0.5):
    """
    Compute precision and recall across the entire dataset.
    
    Args:
        all_predictions: List of dicts with 'boxes', 'labels', 'scores'
        all_ground_truths: List of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        score_threshold: Confidence threshold for predictions
    
    Returns:
        precision, recall, f1, per_class_metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Per-class tracking
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    
    for pred, gt in zip(all_predictions, all_ground_truths):
        # Filter predictions by score threshold
        if len(pred['scores']) > 0:
            mask = pred['scores'] >= score_threshold
            pred_boxes = pred['boxes'][mask]
            pred_labels = pred['labels'][mask]
            pred_scores = pred['scores'][mask]
        else:
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']
        
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        
        # Convert to corner format for IoU computation
        if len(pred_boxes) > 0:
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        else:
            pred_boxes_xyxy = pred_boxes
            
        if len(gt_boxes) > 0:
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        else:
            gt_boxes_xyxy = gt_boxes
        
        # Match predictions to ground truth
        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
            pred_boxes_xyxy, pred_labels, pred_scores,
            gt_boxes_xyxy, gt_labels, iou_threshold
        )
        
        total_tp += len(matches)
        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)
        
        # Per-class accounting
        for pred_idx, gt_idx, _ in matches:
            cls = pred_labels[pred_idx].item()
            class_tp[cls] += 1
        
        for pred_idx in unmatched_preds:
            cls = pred_labels[pred_idx].item()
            class_fp[cls] += 1
        
        for gt_idx in unmatched_gts:
            cls = gt_labels[gt_idx].item()
            class_fn[cls] += 1
    
    # Compute overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute per-class metrics
    class_names = ['Unknown', 'Person', 'Car', 'Vehicle', 'Object', 'Bike']
    per_class_metrics = {}
    
    for cls in range(Config.NUM_CLASSES):
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        per_class_metrics[class_names[cls]] = {
            'precision': p,
            'recall': r,
            'f1': f,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return precision, recall, f1, per_class_metrics


def compute_ap_per_class(all_predictions, all_ground_truths, 
                          class_id, iou_threshold=0.5):
    """
    Compute Average Precision for a single class.
    
    Uses the 11-point interpolation method.
    
    Args:
        all_predictions: List of prediction dicts
        all_ground_truths: List of ground truth dicts
        class_id: Class ID to compute AP for
        iou_threshold: IoU threshold for matching
    
    Returns:
        ap: Average Precision for this class
    """
    # Collect all predictions and ground truths for this class
    all_pred_scores = []
    all_pred_matched = []
    total_gt = 0
    
    for img_idx, (pred, gt) in enumerate(zip(all_predictions, all_ground_truths)):
        # Filter by class
        pred_mask = pred['labels'] == class_id
        gt_mask = gt['labels'] == class_id
        
        pred_boxes = pred['boxes'][pred_mask]
        pred_scores = pred['scores'][pred_mask]
        gt_boxes = gt['boxes'][gt_mask]
        
        total_gt += len(gt_boxes)
        
        if len(pred_boxes) == 0:
            continue
        
        # Convert to corner format
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        
        if len(gt_boxes) > 0:
            gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
            
            # Compute IoU matrix
            iou_matrix = compute_iou_matrix(pred_boxes_xyxy, gt_boxes_xyxy)
            
            # Track which GTs are already matched
            gt_matched = [False] * len(gt_boxes)
            
            # Sort by score
            sorted_indices = torch.argsort(pred_scores, descending=True)
            
            for pred_idx in sorted_indices.tolist():
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx in range(len(gt_boxes)):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = iou_matrix[pred_idx, gt_idx].item()
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    all_pred_scores.append(pred_scores[pred_idx].item())
                    all_pred_matched.append(True)
                    gt_matched[best_gt_idx] = True
                else:
                    all_pred_scores.append(pred_scores[pred_idx].item())
                    all_pred_matched.append(False)
        else:
            # No ground truth - all predictions are false positives
            for score in pred_scores.tolist():
                all_pred_scores.append(score)
                all_pred_matched.append(False)
    
    if total_gt == 0:
        return 0.0
    
    # Sort by score (descending)
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    all_pred_matched = [all_pred_matched[i] for i in sorted_indices]
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(all_pred_matched)
    fp_cumsum = np.cumsum([not m for m in all_pred_matched])
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt
    
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        # Find max precision at recall >= t
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 11
    
    return ap


def compute_map(all_predictions, all_ground_truths, iou_threshold=0.5):
    """
    Compute mean Average Precision across all classes.
    
    Args:
        all_predictions: List of prediction dicts
        all_ground_truths: List of ground truth dicts
        iou_threshold: IoU threshold for matching
    
    Returns:
        mAP: mean Average Precision
        per_class_ap: dict of AP per class
    """
    class_names = ['Unknown', 'Person', 'Car', 'Vehicle', 'Object', 'Bike']
    per_class_ap = {}
    
    for cls in range(Config.NUM_CLASSES):
        ap = compute_ap_per_class(all_predictions, all_ground_truths, cls, iou_threshold)
        per_class_ap[class_names[cls]] = ap
    
    # mAP is the mean of APs for classes that have ground truth
    valid_aps = [ap for ap in per_class_ap.values() if ap > 0]
    mAP = np.mean(valid_aps) if valid_aps else 0.0
    
    return mAP, per_class_ap


# =============================================================================
# SECTION 4: Inference Function
# =============================================================================

def run_inference(model, dataloader, device, score_threshold=0.0):
    """
    Run inference on the entire dataset.
    
    Args:
        model: Trained MovedObjectDETR model
        dataloader: DataLoader for test set
        device: torch device
        score_threshold: Minimum score to keep predictions
    
    Returns:
        all_predictions: List of dicts with 'boxes', 'labels', 'scores'
        all_ground_truths: List of dicts with 'boxes', 'labels'
        all_image_pairs: List of (img1, img2) tensors for visualization
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_image_pairs = []
    
    with torch.no_grad():
        for batch_idx, (img1, img2, targets) in enumerate(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Forward pass
            outputs = model(img1, img2)
            
            # Process each image in batch
            batch_size = img1.shape[0]
            
            for i in range(batch_size):
                # Get predictions for this image
                logits = outputs.logits[i]  # [num_queries, num_classes + 1]
                pred_boxes = outputs.pred_boxes[i]  # [num_queries, 4]
                
                # Get predicted classes and scores
                probs = logits.softmax(-1)
                # Last class is "no object" - exclude it
                scores, labels = probs[:, :-1].max(-1)
                
                # Filter by score threshold
                mask = scores > score_threshold
                
                pred_dict = {
                    'boxes': pred_boxes[mask].cpu(),
                    'labels': labels[mask].cpu(),
                    'scores': scores[mask].cpu()
                }
                
                gt_dict = {
                    'boxes': targets[i]['boxes'],
                    'labels': targets[i]['labels'] if 'labels' in targets[i] else targets[i]['class_labels']
                }
                
                all_predictions.append(pred_dict)
                all_ground_truths.append(gt_dict)
                all_image_pairs.append((img1[i].cpu(), img2[i].cpu()))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    return all_predictions, all_ground_truths, all_image_pairs


# =============================================================================
# SECTION 5: Visualization
# =============================================================================

def visualize_predictions(img1, img2, pred_boxes, pred_labels, pred_scores,
                          gt_boxes, gt_labels, save_path, 
                          score_threshold=0.5, class_names=None):
    """
    Create visualization comparing predictions to ground truth.
    
    Args:
        img1: First image tensor [C, H, W]
        img2: Second image tensor [C, H, W]
        pred_boxes: Predicted boxes [N, 4] in (cx, cy, w, h) normalized
        pred_labels: Predicted labels [N]
        pred_scores: Prediction scores [N]
        gt_boxes: Ground truth boxes [M, 4] in (cx, cy, w, h) normalized
        gt_labels: Ground truth labels [M]
        save_path: Where to save the visualization
        score_threshold: Minimum score to display
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['Unknown', 'Person', 'Car', 'Vehicle', 'Object', 'Bike']
    
    # Convert tensors to PIL images
    def tensor_to_pil(t):
        # Denormalize if needed (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = t * std + mean
        t = t.clamp(0, 1)
        return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    img1_pil = tensor_to_pil(img1)
    img2_pil = tensor_to_pil(img2)
    
    # Get image dimensions
    W, H = img2_pil.size
    
    # Create side-by-side image
    combined = Image.new('RGB', (W * 2, H))
    combined.paste(img1_pil, (0, 0))
    combined.paste(img2_pil, (W, 0))
    
    draw = ImageDraw.Draw(combined)
    
    # Color scheme: Green for GT, Red for predictions
    colors = {
        'gt': (0, 255, 0),      # Green
        'pred': (255, 0, 0),    # Red
        'matched': (0, 0, 255)  # Blue for matched predictions
    }
    
    # Draw ground truth boxes on img2 (right side)
    for box, label in zip(gt_boxes, gt_labels):
        cx, cy, w, h = box.tolist()
        x1 = int((cx - w/2) * W) + W  # Offset for right image
        y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W) + W
        y2 = int((cy + h/2) * H)
        
        draw.rectangle([x1, y1, x2, y2], outline=colors['gt'], width=2)
        label_text = f"GT: {class_names[label.item()]}"
        draw.text((x1, y1 - 15), label_text, fill=colors['gt'])
    
    # Draw predictions on img2 (right side)
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < score_threshold:
            continue
        
        cx, cy, w, h = box.tolist()
        x1 = int((cx - w/2) * W) + W
        y1 = int((cy - h/2) * H)
        x2 = int((cx + w/2) * W) + W
        y2 = int((cy + h/2) * H)
        
        draw.rectangle([x1, y1, x2, y2], outline=colors['pred'], width=2)
        label_text = f"{class_names[label.item()]}: {score:.2f}"
        draw.text((x1, y2 + 2), label_text, fill=colors['pred'])
    
    # Add legend
    draw.text((10, 10), "Green = Ground Truth", fill=colors['gt'])
    draw.text((10, 25), "Red = Predictions", fill=colors['pred'])
    draw.text((10, 40), f"Score threshold: {score_threshold}", fill=(255, 255, 255))
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.save(save_path)


def generate_visualizations(all_predictions, all_ground_truths, all_image_pairs,
                            output_dir, num_samples=10, score_threshold=0.5):
    """
    Generate visualizations for a sample of the test set.
    
    Args:
        all_predictions: List of prediction dicts
        all_ground_truths: List of ground truth dicts
        all_image_pairs: List of (img1, img2) tensors
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        score_threshold: Minimum score to display
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select samples (evenly spaced)
    total = len(all_predictions)
    indices = np.linspace(0, total - 1, min(num_samples, total), dtype=int)
    
    for idx in indices:
        pred = all_predictions[idx]
        gt = all_ground_truths[idx]
        img1, img2 = all_image_pairs[idx]
        
        save_path = os.path.join(output_dir, f'sample_{idx:04d}.png')
        
        visualize_predictions(
            img1, img2,
            pred['boxes'], pred['labels'], pred['scores'],
            gt['boxes'], gt['labels'],
            save_path, score_threshold
        )
    
    print(f"Saved {len(indices)} visualizations to {output_dir}")


# =============================================================================
# SECTION 6: Main Evaluation Function
# =============================================================================

def evaluate(checkpoint_path, output_dir=None, score_threshold=0.5, 
             iou_threshold=0.5, num_vis=10):
    """
    Main evaluation function.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_dir: Where to save results (default: Config.VIS_DIR)
        score_threshold: Minimum score for predictions
        iou_threshold: IoU threshold for matching
        num_vis: Number of visualizations to generate
    
    Returns:
        results: Dict containing all metrics
    """
    if output_dir is None:
        output_dir = Config.VIS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load model
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading model...")
    
    model = MovedObjectDETR(num_classes=Config.NUM_CLASSES)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Training loss was: {checkpoint.get('loss', 'unknown')}")
    
    # -------------------------------------------------------------------------
    # Step 2: Prepare test data
    # -------------------------------------------------------------------------
    print("\n[2/5] Preparing test data...")
    
    # Load test split
    _, test_files = MovedObjectDataset.load_split_files('data')
    print(f"  Test set size: {len(test_files)} samples")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((Config.DETR_INPUT_SIZE, Config.DETR_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    test_dataset = MovedObjectDataset(
        annotation_dir=Config.ANNOTATION_DIR,
        image_base_dir=Config.DATA_BASE_DIR,
        file_list=test_files,
        transform=transform
    )
    
    # Import collate_fn from train.py
    from train import collate_fn
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Run inference
    # -------------------------------------------------------------------------
    print("\n[3/5] Running inference...")
    
    all_predictions, all_ground_truths, all_image_pairs = run_inference(
        model, test_loader, device, score_threshold=0.0
    )
    
    print(f"  Processed {len(all_predictions)} images")
    
    # -------------------------------------------------------------------------
    # Step 4: Compute metrics
    # -------------------------------------------------------------------------
    print("\n[4/5] Computing metrics...")
    
    # Precision, Recall, F1
    precision, recall, f1, per_class_metrics = compute_precision_recall(
        all_predictions, all_ground_truths, 
        iou_threshold=iou_threshold, 
        score_threshold=score_threshold
    )
    
    # mAP
    mAP, per_class_ap = compute_map(
        all_predictions, all_ground_truths, 
        iou_threshold=iou_threshold
    )
    
    # Also compute mAP at different IoU thresholds
    mAP_50, _ = compute_map(all_predictions, all_ground_truths, iou_threshold=0.5)
    mAP_75, _ = compute_map(all_predictions, all_ground_truths, iou_threshold=0.75)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics (IoU={iou_threshold}, Score={score_threshold}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  mAP@0.5:   {mAP_50:.4f}")
    print(f"  mAP@0.75:  {mAP_75:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AP':<10}")
    print("-" * 60)
    
    class_names = ['Unknown', 'Person', 'Car', 'Vehicle', 'Object', 'Bike']
    for cls_name in class_names:
        m = per_class_metrics.get(cls_name, {})
        ap = per_class_ap.get(cls_name, 0)
        print(f"{cls_name:<12} {m.get('precision', 0):<10.4f} {m.get('recall', 0):<10.4f} "
              f"{m.get('f1', 0):<10.4f} {ap:<10.4f}")
    
    print("-" * 60)
    
    # -------------------------------------------------------------------------
    # Step 5: Generate visualizations
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating visualizations...")
    
    vis_dir = os.path.join(output_dir, 'predictions')
    generate_visualizations(
        all_predictions, all_ground_truths, all_image_pairs,
        vis_dir, num_samples=num_vis, score_threshold=score_threshold
    )
    
    # -------------------------------------------------------------------------
    # Save results to JSON
    # -------------------------------------------------------------------------
    results = {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP_50': mAP_50,
            'mAP_75': mAP_75
        },
        'per_class': per_class_metrics,
        'per_class_ap': per_class_ap,
        'settings': {
            'checkpoint': checkpoint_path,
            'score_threshold': score_threshold,
            'iou_threshold': iou_threshold,
            'num_test_samples': len(all_predictions)
        }
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


# =============================================================================
# SECTION 7: Command-line Interface
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MovedObjectDETR model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Minimum confidence score for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for matching predictions to GT')
    parser.add_argument('--num_vis', type=int, default=10,
                        help='Number of visualizations to generate')
    
    args = parser.parse_args()
    
    results = evaluate(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        num_vis=args.num_vis
    )