'''
Custom PyTorch Dataset class
Reads matched annotation files
Loads image pairs
Applies transformations/preprocessing
Implements train/test split
Returns: image1, image2, bounding boxes, labels
'''
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

#Constants for image dimensions
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


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
                
        #Step 4: Parse annotation file
        boxes = []
        labels = []
        
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        
        #Loop through odd-indexed lines (new positions only)
        for i in range(1, len(lines), 2):
            parts = lines[i].strip().split()
            
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            obj_type = int(parts[5])
            
            boxes.append([x, y, w, h])
            labels.append(obj_type)
        
        #Step 5: Convert to DETR format
        detr_boxes = []
        for box in boxes:
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
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        
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
        



        