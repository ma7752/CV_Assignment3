'''
Helper functions for visualization
Bounding box utilities (IoU calculation, format conversion)
Logging utilities
Any other reusable functions
'''
import os
import random
from sklearn.model_selection import train_test_split

def create_train_test_split(annotation_dir, train_split=0.8, random_seed=42):
    """
    Split annotation files into train and test sets.
    Args:
        annotation_dir: Path to matched annotations folder
        train_split: Fraction of data for training (default 0.8 for 80%)   
        random_seed: Random seed for reproducibility
    Returns:
        train_files: List of training annotation filenames
        test_files: List of test annotation filenames
    """
    # Get all annotation files
    all_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
    # Split
    train_files, test_files = train_test_split(
        all_files,
        train_size=train_split,
        random_state=random_seed,
        shuffle=True
    )
    print(f"Total annotations: {len(all_files)}")
    print(f"Training set: {len(train_files)} ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"Test set: {len(test_files)} ({len(test_files)/len(all_files)*100:.1f}%)")
    return train_files, test_files

def save_split_files(train_files, test_files, output_dir='data'):
    """
    Save train/test split to text files for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train_files.txt'), 'w') as f:      
        f.write('\n'.join(train_files))
    with open(os.path.join(output_dir, 'test_files.txt'), 'w') as f:       
        f.write('\n'.join(test_files))
    print(f"Split files saved to {output_dir}/")

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

# Test this function
if __name__ == "__main__":
    train_files, test_files = create_train_test_split('data/matched_annotations')
    save_split_files(train_files, test_files)