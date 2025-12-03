'''
Custom PyTorch Dataset class
Reads matched annotation files
Loads image pairs
Applies transformations/preprocessing
Implements train/test split
Returns: image1, image2, bounding boxes, labels

IMPORTANT: Per professor's clarification, we only detect objects that:
  1. Have IoU = 0 between old and new positions (object moved completely)
  2. Have IoU > 0 but different classes (class changed - rare)
  
Objects with IoU > 0 AND same class are NOT considered "moved" and should not be detected.
'''
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

#Constants for image dimensions
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


def compute_iou_single(box1, box2):
    """
    Compute IoU between two boxes in [x, y, w, h] format (pixel coordinates).
    
    Args:
        box1: [x, y, w, h] - top-left corner + width/height
        box2: [x, y, w, h] - top-left corner + width/height
    
    Returns:
        IoU value (float)
    """
    # Convert to corner format (x1, y1, x2, y2)
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    
    # Union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


class MovedObjectDataset(Dataset):
    def __init__(self, annotation_dir, image_base_dir, file_list=None, transform=None, **kwargs):
        """
        Args:
            annotation_dir: Path to matched_annotations folder
            image_base_dir: Path to cv_data_hw2/data folder
            file_list: Optional list of specific annotation files to use (for train/test split)
            transform: Optional transforms to apply to images
        """
        self.annotation_dir = annotation_dir
        self.image_base_dir = image_base_dir
        self.transform = transform
        
        # Get list of all annotation files or use provided list
        if file_list is not None:
            self.annotation_files = file_list
        else:
            self.annotation_files = [
                f for f in os.listdir(annotation_dir)
                if f.endswith('.txt')
            ]
    
    def __len__(self):
        return len(self.annotation_files)
    
    def __getitem__(self, idx):
        #Step 1: Get annotation filename
        ann_filename = self.annotation_files[idx]
        ann_path = os.path.join(self.annotation_dir, ann_filename)
        
        #Step 2: Parse filename to get image paths
        base_name = ann_filename.replace('.txt', '')
        parts = base_name.split('-')
        
        folder_name = parts[0]
        img1_name = parts[1] + '.png'
        img2_name = parts[2].replace('_match', '') + '.png'
        
        img1_path = os.path.join(self.image_base_dir, folder_name, img1_name)
        img2_path = os.path.join(self.image_base_dir, folder_name, img2_name)
        
        #Step 3: Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')



        # Get image dimensions dynamically
        img_width, img_height = img1.size 
        img2_width, img2_height = img2.size

        # Sanity check - both images should have same dimensions
        assert img_width == img2_width and img_height == img2_height, \
            f"Image size mismatch: img1={img1.size}, img2={img2.size}"
                
        #Step 4: Parse annotation file and filter for TRULY MOVED objects
        # Per professor's clarification:
        #   - Detect if IoU = 0 (object moved completely)
        #   - Detect if IoU > 0 but classes differ (class changed)
        #   - Do NOT detect if IoU > 0 AND same class (object didn't move)
        
        moved_boxes = []  # Boxes in img2 (new positions) for moved objects
        moved_labels = []
        
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        
        # Process pairs of lines: even = old position (img1), odd = new position (img2)
        for i in range(0, len(lines) - 1, 2):
            # Parse OLD position (in img1)
            old_parts = lines[i].strip().split()
            old_x = float(old_parts[1])
            old_y = float(old_parts[2])
            old_w = float(old_parts[3])
            old_h = float(old_parts[4])
            old_class = int(old_parts[5])
            old_box = [old_x, old_y, old_w, old_h]
            
            # Parse NEW position (in img2)
            new_parts = lines[i + 1].strip().split()
            new_x = float(new_parts[1])
            new_y = float(new_parts[2])
            new_w = float(new_parts[3])
            new_h = float(new_parts[4])
            new_class = int(new_parts[5])
            new_box = [new_x, new_y, new_w, new_h]
            
            # Compute IoU between old and new positions
            iou = compute_iou_single(old_box, new_box)
            
            # Determine if this is a "moved" object we should detect
            is_moved = False
            
            if iou == 0:
                # Case 1: No overlap - object moved completely
                is_moved = True
            elif old_class != new_class:
                # Case 2: Overlap exists but class changed
                is_moved = True
            # else: iou > 0 AND same class → NOT moved, skip
            
            if is_moved:
                moved_boxes.append(new_box)
                moved_labels.append(new_class)
        
        #Step 5: Convert to DETR format (only for moved objects)
        detr_boxes = []
        for box in moved_boxes:
            x, y, w, h = box
            
            # Convert to center format
            x_center = x + w/2
            y_center = y + h/2
            
            # Normalize
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            detr_boxes.append([x_center_norm, y_center_norm, w_norm, h_norm])
        
        #Step 6: Convert to tensors
        boxes_tensor = torch.as_tensor(detr_boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(moved_labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }
        
        #Step 7: Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, target
    
    def load_split_files(data_dir='data'):
        """
        Load previously saved train/test split.
        
        Returns:
            train_files: List of training annotation filenames
            test_files: List of test annotation filenames
        """
        train_path = os.path.join(data_dir, 'train_files.txt')
        test_path = os.path.join(data_dir, 'test_files.txt')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Split files not found. Run create_train_test_split first.")
        
        with open(train_path, 'r') as f:
            train_files = [line.strip() for line in f if line.strip()]
        
        with open(test_path, 'r') as f:
            test_files = [line.strip() for line in f if line.strip()]
        
        return train_files, test_files
        



        