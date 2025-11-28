import sys
sys.path.append('.')

from src.dataloader import MovedObjectDataset
import torchvision.transforms as transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = MovedObjectDataset(
    annotation_dir='data/matched_annotations',
    image_base_dir='/mnt/c/Users/muham/OneDrive/Desktop/ComputerVision/cv_data_hw2/data',
    transform=transform
)

print(f"Dataset size: {len(dataset)}")

# Test first sample
print("\n=== Testing Sample 0 ===")
img1, img2, target = dataset[0]

print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")
print(f"Number of objects: {len(target['labels'])}")
print(f"Labels: {target['labels']}")

print("\n=== Raw Annotation ===")
ann_file = 'data/matched_annotations/' + dataset.annotation_files[0]
with open(ann_file, 'r') as f:
    lines = f.readlines()
    print(f"Line 0 (old pos): {lines[0].strip()}")
    print(f"Line 1 (new pos): {lines[1].strip()}")

print("\n=== Computed Boxes (normalized) ===")
for i, box in enumerate(target['boxes']):
    x_c, y_c, w, h = box
    print(f"Object {i}: x_center={x_c:.4f}, y_center={y_c:.4f}, w={w:.4f}, h={h:.4f}")