import torch
import torch.nn as nn
from transformers import DetrForObjectDetection


class FeatureDifferenceLayer(nn.Module):
    """
    Custom layer that replaces ResNet's layer4 for change detection.
    
    Traditional ResNet layer4:
        - Takes 1024-channel features from layer3
        - Applies several residual blocks
        - Outputs 2048-channel features
    
    Our FeatureDifferenceLayer:
        - Takes 1024-channel features from layer3 (from TWO images)
        - Computes feature difference: feat2 - feat1
        - Projects difference to 2048 channels using 1x1 convolution
        - This highlights what changed between the two images!
    
    Why this works:
        - Feature differences emphasize regions that changed
        - DETR's transformer can focus on these changed regions
        - We skip the computational cost of layer4
        - The 1x1 projection is much faster than full layer4
    
    Shape flow:
        img1 → layer3 → [B, 1024, H/16, W/16] → (stored)
        img2 → layer3 → [B, 1024, H/16, W/16] → difference → [B, 1024, H/16, W/16]
        difference → 1x1 conv → [B, 2048, H/16, W/16] → continues to DETR
    """
    def __init__(self):
        super().__init__()
        # 1x1 convolution to project 1024 → 2048 channels
        # This replaces the channel expansion that layer4 would normally do
        self.proj = nn.Conv2d(1024, 2048, kernel_size=1, bias=True)
        
        # Temporary storage for first image's features
        # Will be set to None after each pair is processed
        self.feat1 = None
        
    def forward(self, x):
        """
        Forward pass with stateful behavior.
        
        Call 1 (img1): Store features, return projected features (discarded later)
        Call 2 (img2): Compute difference with stored feat1, project, return
        
        Args:
            x: Feature map from layer3, shape [B, 1024, H/16, W/16]
        
        Returns:
            Projected features, shape [B, 2048, H/16, W/16]
        """
        if self.feat1 is None:
            # FIRST CALL (img1): Store features for later comparison
            self.feat1 = x
            # Return projected features (this output will be discarded)
            # We project here just to maintain consistent output shape
            return self.proj(x)
        else:
            # SECOND CALL (img2): Compute difference and project
            # This is the key step: feat_diff highlights what changed!
            feat_diff = x - self.feat1  # [B, 1024, H/16, W/16]
            
            # Reset state for next image pair
            self.feat1 = None
            
            # Project to 2048 channels to match expected input to DETR
            return self.proj(feat_diff)  # [B, 2048, H/16, W/16]


class MovedObjectDETR(nn.Module):
    """
    DETR-based object detection model for change detection / moved object detection.
    
    Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Input: Two images (img1, img2) - same scene at different times │
    └─────────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ ResNet Backbone (up to layer3) - FROZEN                         │
    │   • Extracts features from both images                          │
    │   • Output: 1024 channels at 1/16 resolution                    │
    └─────────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ FeatureDifferenceLayer (replaces layer4) - TRAINABLE            │
    │   • Computes feat2 - feat1                                      │
    │   • Projects 1024 → 2048 channels                               │
    │   • Highlights changed regions                                  │
    └─────────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ DETR Pipeline - TRAINABLE                                       │
    │   • Input projection: 2048 → 256 dims                           │
    │   • Positional encoding                                         │
    │   • Transformer encoder (6 layers)                              │
    │   • Transformer decoder (6 layers, 100 object queries)          │
    │   • Classification head: 256 → num_classes                      │
    │   • Bounding box head: 256 → 4 (cx, cy, w, h)                   │
    └─────────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │ Output: 100 object predictions                                  │
    │   • logits: [B, 100, num_classes+1] - class probabilities       │
    │   • pred_boxes: [B, 100, 4] - normalized box coordinates        │
    └─────────────────────────────────────────────────────────────────┘
    
    Key Innovation:
        Instead of processing one image, we process TWO images and compute
        feature differences. This allows the model to focus on objects that
        moved, appeared, or disappeared between the two frames.
    
    Training Strategies (from assignment):
        1. Fine-tune all parameters (except frozen backbone)
        2. Fine-tune only transformer classification head
        3. Fine-tune only transformer block (encoder + decoder)
        4. Compare performance across strategies
    """
    
    def __init__(self, num_classes=6):
        """
        Initialize the MovedObjectDETR model.
        
        Args:
            num_classes: Number of object classes (default 6 for VIRAT dataset)
                        Classes: Unknown=0, person=1, car=2, vehicle=3, object=4, bike=5
                        
        Architecture modifications:
            1. Load pretrained DETR-ResNet50
            2. Replace layer4 with FeatureDifferenceLayer
            3. Freeze ResNet backbone (stem + layer1-3)
            4. Adjust classification head for our num_classes
        """
        super().__init__()
        
        # Load pretrained DETR with ResNet-50 backbone
        # This gives us strong pretrained weights from COCO dataset
        self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # Access the ResNet backbone inside DETR
        # Path: self.detr.model.backbone.conv_encoder.model = ResNet
        backbone = self.detr.model.backbone.conv_encoder.model
        
        # Replace layer4 with our custom feature difference layer
        # Original layer4: several ResNet blocks (1024 → 2048 channels)
        # Our layer: computes difference + 1x1 projection (1024 → 2048 channels)
        self.custom_layer4 = FeatureDifferenceLayer()
        backbone.layer4 = self.custom_layer4
        
        # Freeze all backbone parameters EXCEPT our custom layer4
        # This preserves pretrained features and reduces training time
        for name, param in backbone.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False
        
        # Adjust the classification head for our dataset
        # DETR predicts num_classes + 1 (extra class for "no object")
        # Original DETR: 92 classes (91 COCO classes + no-object)
        # Our model: num_classes + 1
        num_labels = num_classes + 1
        self.detr.class_labels_classifier = nn.Linear(
            self.detr.class_labels_classifier.in_features,  # 256 (transformer hidden dim)
            num_labels
        )
        
        print("Model architecture created:")
        print("  - ResNet up to layer3 (frozen)")
        print("  - Custom layer4: computes feature difference + projection (trainable)")
        print("  - DETR transformer + heads (trainable)")
        print(f"  - Classification: {num_labels} classes ({num_classes} objects + no-object)")
        
        # Print parameter statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"\nParameter counts:")
        print(f"  - Total: {total_params:,}")
        print(f"  - Trainable: {trainable_params:,}")
        print(f"  - Frozen: {frozen_params:,}")
        print(f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    
    def forward(self, img1, img2, targets=None):
        """
        Forward pass through the entire model.
        
        This method orchestrates the two-image processing:
        1. Pass img1 through backbone → features stored in layer4
        2. Pass img2 through backbone → difference computed in layer4
        3. Continue through DETR's transformer and prediction heads
        
        Args:
            img1: First image tensor, shape [B, 3, H, W]
                  Typically the "before" image or reference frame
            img2: Second image tensor, shape [B, 3, H, W]
                  Typically the "after" image or query frame
            targets: Optional list of target dictionaries for training
                    Each dict should contain:
                        - 'class_labels': [num_objects] - class indices
                        - 'boxes': [num_objects, 4] - boxes in [cx, cy, w, h] format (normalized)
        
        Returns:
            Dictionary containing:
                - 'logits': [B, 100, num_classes+1] - class predictions for each query
                - 'pred_boxes': [B, 100, 4] - predicted boxes in [cx, cy, w, h] format
                - 'loss' (if targets provided): scalar total loss
                - Additional loss components (if targets provided)
        
        Processing flow:
            img1 [B,3,H,W]
              ↓
            ResNet (stem + layer1-3)
              ↓
            feat1 [B,1024,H/16,W/16] → stored in self.custom_layer4.feat1
              ↓
            (discard output)
            
            img2 [B,3,H,W]
              ↓
            ResNet (stem + layer1-3)
              ↓
            feat2 [B,1024,H/16,W/16]
              ↓
            custom_layer4: difference = feat2 - feat1
              ↓
            proj(difference) [B,2048,H/16,W/16]
              ↓
            DETR pipeline (projection → transformer → heads)
              ↓
            predictions {logits, boxes}
        """
        batch_size = img1.shape[0]
        device = img1.device
        
        # Create pixel masks (all ones = all pixels are valid, no padding)
        # DETR uses masks to ignore padded regions, but we don't have any padding
        h, w = img1.shape[2], img1.shape[3]
        pixel_mask = torch.ones((batch_size, h, w), dtype=torch.long, device=device)
        
        # STEP 1: Pass img1 through backbone
        # This will go through: stem → layer1 → layer2 → layer3 → custom_layer4
        # The custom_layer4 will STORE feat1 and return projected features
        # We discard this output because it's not the difference yet
        _ = self.detr.model.backbone(img1, pixel_mask)
        
        # STEP 2: Pass img2 through DETR's full pipeline
        # This will:
        #   1. Go through backbone: stem → layer1 → layer2 → layer3 → custom_layer4
        #   2. custom_layer4 computes DIFFERENCE (feat2 - feat1) and projects it
        #   3. Continue through DETR: projection → transformer → classification/bbox heads
        #   4. If targets provided, compute loss using Hungarian matching
        outputs = self.detr(pixel_values=img2, pixel_mask=pixel_mask, labels=targets)
        
        return outputs



if __name__ == "__main__":
    model = MovedObjectDETR(num_classes=6)
    
    img1 = torch.randn(2, 3, 800, 800)
    img2 = torch.randn(2, 3, 800, 800)
    
    print("\nTesting forward pass...")
    outputs = model(img1, img2)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Pred boxes shape: {outputs['pred_boxes'].shape}")
    print("\nModel test successful!")