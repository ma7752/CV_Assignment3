# CV Assignment 3 - Development Log

## Project Overview
**Goal:** Implement a DETR-based model for moved object detection using image pairs from the VIRAT dataset.

**Architecture (Option 1):**
- ResNet backbone up to layer3 (frozen) → Extract features from both images
- Custom layer4: Compute feature difference + projection (trainable)
- DETR transformer + prediction heads (trainable)
- Classification: 7 classes (6 object classes + 1 "no-object" class)

**Dataset:** VIRAT dataset with 370 image pairs and 18 annotation files (14 train, 4 test)

---

## Issues Encountered and Solutions

### Issue 1: HuggingFace DETR API Complexity

**Problem:**
Initial attempts to manually call DETR's encoder and decoder led to complex errors:
```
UnboundLocalError: cannot access local variable 'hidden_states_original'
```

We tried to manually orchestrate the DETR pipeline:
```python
# BROKEN APPROACH - Too complex
encoder_outputs = self.detr.model.encoder(...)
decoder_outputs = self.detr.model.decoder(...)
```

**Solution:**
Instead of manually calling encoder/decoder, we simply **replaced layer4** in the backbone and let DETR handle everything else:
```python
# WORKING APPROACH - Simple layer replacement
backbone = self.detr.model.backbone.conv_encoder.model
backbone.layer4 = self.custom_layer4  # Our FeatureDifferenceLayer

# Then just call DETR normally - it handles the rest
outputs = self.detr(pixel_values=img2, pixel_mask=pixel_mask, labels=targets)
```

**Key Insight:** DETR's forward pass automatically uses whatever is in `backbone.layer4`. By replacing it with our custom layer, we inject our feature difference computation into the existing pipeline.

---

### Issue 2: Stateful Feature Difference Layer

**Problem:**
We need to compute the difference between features from two images (img1 and img2), but DETR's backbone only processes one image at a time.

**Solution:**
Created a **stateful** `FeatureDifferenceLayer` that:
1. On first call (img1): Stores features in `self.feat1` and returns projected output
2. On second call (img2): Computes `feat2 - feat1`, clears state, and returns projected difference

```python
class FeatureDifferenceLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat1 = None  # Temporary storage for first image features
        self.proj = nn.Conv2d(1024, 2048, kernel_size=1)  # Project to expected size
    
    def forward(self, x):
        if self.feat1 is None:
            # FIRST CALL: Store features
            self.feat1 = x.clone()
            return self.proj(x)  # Return something valid
        else:
            # SECOND CALL: Compute difference
            feat_diff = x - self.feat1
            self.feat1 = None  # Reset for next pair
            return self.proj(feat_diff)
```

**Usage in forward pass:**
```python
def forward(self, img1, img2, targets=None):
    # Step 1: Process img1 → stores features in layer4
    _ = self.detr.model.backbone(img1, pixel_mask)
    
    # Step 2: Process img2 through full DETR → computes difference in layer4
    outputs = self.detr(pixel_values=img2, pixel_mask=pixel_mask, labels=targets)
    return outputs
```

---

### Issue 3: Target Format - 'labels' vs 'class_labels'

**Problem:**
```
KeyError: 'labels'
```

Our dataloader used `'labels'` key, but HuggingFace DETR expects `'class_labels'`.

**Solution:**
Created a flexible conversion function that handles both formats:
```python
def convert_targets_to_detr_format(targets):
    if targets is None:
        return None
    
    detr_targets = []
    for target in targets:
        if 'class_labels' in target:
            # Already in DETR format
            detr_target = {'class_labels': target['class_labels'], 'boxes': target['boxes']}
        elif 'labels' in target:
            # Convert from our dataloader format
            detr_target = {'class_labels': target['labels'], 'boxes': target['boxes']}
        else:
            raise KeyError(f"Target must contain 'labels' or 'class_labels', got: {target.keys()}")
        detr_targets.append(detr_target)
    return detr_targets
```

---

### Issue 4: Class Weight Tensor Mismatch (CRITICAL)

**Problem:**
```
RuntimeError: weight tensor should be defined either for all or no classes
```

This was the most complex issue. The error occurred in DETR's loss function at:
```python
# In transformers/loss/loss_for_object_detection.py
loss_ce = nn.functional.cross_entropy(
    source_logits.transpose(1, 2), 
    target_classes, 
    self.empty_weight  # ← This tensor had wrong size!
)
```

**Root Cause:**
DETR creates an `empty_weight` buffer during model initialization with the original 92 classes. When we changed the classifier head to 7 classes AFTER loading, the loss function's weight tensor still had size 92.

**Failed Attempts:**
1. Setting `self.detr.config.num_labels = 7` after loading → Didn't update the loss weights
2. Creating new `class_labels_classifier` layer → Classifier worked, but loss weights wrong
3. Trying to register new `empty_weight` buffer on model → Loss class has its own buffer

**Working Solution:**
Configure `num_labels` in `DetrConfig` **BEFORE** loading the model:
```python
from transformers import DetrForObjectDetection, DetrConfig

# Configure BEFORE loading
config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
config.num_labels = 7  # 6 classes + 1 no-object
config.eos_coefficient = 0.1  # Weight for "no object" class

# Load with custom config - loss weights will be correct size
self.detr = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    config=config,
    ignore_mismatched_sizes=True  # Allow classifier size mismatch
)
```

**Key Insight:** The `ignore_mismatched_sizes=True` flag allows loading pretrained weights even when some layer sizes don't match. The mismatched layers (classifier) are randomly initialized.

---

### Issue 5: Test File Organization

**Problem:**
Multiple test files scattered across the project with inconsistent naming and some broken tests.

**Solution:**
Organized tests into a proper pytest structure:
```
tests/
├── __init__.py
├── conftest.py      # Shared fixtures (dummy_images, dummy_targets)
├── test_model.py    # Unit tests for FeatureDifferenceLayer and MovedObjectDETR
├── test_config.py   # Configuration validation tests
└── test_dataloader.py  # Dataset loading tests

# Root-level manual test scripts
test_model_detailed.py  # Comprehensive model verification
test_real_data.py       # Real VIRAT data testing
```

---

## Class Numbering Reference

| Index | Class Name    | Description |
|-------|---------------|-------------|
| 0     | Unknown       | Unknown object type |
| 1     | Person        | Human/pedestrian |
| 2     | Car           | Automobile |
| 3     | Other Vehicle | Non-car vehicles |
| 4     | Other Object  | Miscellaneous objects |
| 5     | Bike          | Bicycle/motorcycle |
| 6     | No-Object     | DETR's "no object" class |

**Total:** 6 object classes + 1 no-object = **7 classes in output**

---

## Final Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | 28,660,172 |
| Trainable Parameters | 20,147,468 (70.30%) |
| Frozen Parameters | 8,512,704 (29.70%) |
| Output Queries | 100 (standard DETR) |
| Output Shape (logits) | [batch, 100, 7] |
| Output Shape (boxes) | [batch, 100, 4] |
| Box Format | [cx, cy, w, h] normalized |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | Core model: `MovedObjectDETR`, `FeatureDifferenceLayer`, `convert_targets_to_detr_format` |
| `src/config.py` | Configuration constants (paths, hyperparameters, class mappings) |
| `src/dataloader.py` | VIRAT dataset loading and preprocessing |
| `src/train.py` | Training script (to be implemented) |
| `src/evaluate.py` | Evaluation metrics (to be implemented) |

---

## Lessons Learned

1. **Work WITH frameworks, not against them:** Instead of manually calling DETR's internals, replacing a layer in the backbone was much simpler.

2. **Stateful modules require careful design:** The FeatureDifferenceLayer's state must be properly managed to avoid stale features.

3. **Order matters for configuration:** Setting config BEFORE loading a pretrained model is different from setting it AFTER.

4. **Read error tracebacks carefully:** The "weight tensor" error pointed directly to the loss function, which led us to understand that the loss object has its own buffers.

5. **Test incrementally:** Starting with dummy data before real data helped isolate issues.

---

## Next Steps

- [ ] Test with real VIRAT dataset images
- [ ] Implement training script (`src/train.py`)
- [ ] Implement evaluation metrics (`src/evaluate.py`)
- [ ] Run 4 fine-tuning experiments
- [ ] Generate comparison results and visualizations
