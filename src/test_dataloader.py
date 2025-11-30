import sys
sys.path.append('.')

from src.dataloader import MovedObjectDataset
from src.utils import load_split_files
import torchvision.transforms as transforms

# Load train/test split
train_files, test_files = load_split_files('data')

print(f"Train files: {len(train_files)}")
print(f"Test files: {len(test_files)}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create train dataset
train_dataset = MovedObjectDataset(
    annotation_dir='data/matched_annotations',
    image_base_dir='/mnt/c/Users/muham/OneDrive/Desktop/ComputerVision/cv_data_hw2/data',
    file_list=train_files,
    transform=transform
)

# Create test dataset
test_dataset = MovedObjectDataset(
    annotation_dir='data/matched_annotations',
    image_base_dir='/mnt/c/Users/muham/OneDrive/Desktop/ComputerVision/cv_data_hw2/data',
    file_list=test_files,
    transform=transform
)

print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Test loading one sample from each
print("\n=== Testing Train Sample ===")
img1, img2, target = train_dataset[0]
print(f"Image shapes: {img1.shape}, {img2.shape}")
print(f"Num objects: {len(target['labels'])}")

print("\n=== Testing Test Sample ===")
img1, img2, target = test_dataset[0]
print(f"Image shapes: {img1.shape}, {img2.shape}")
print(f"Num objects: {len(target['labels'])}")