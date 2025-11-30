'''
Main training loop
Load data using dataloader
Forward pass, loss calculation, backpropagation
Save checkpoints
Log training metrics
Implement different fine-tuning strategies
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys

sys.path.append('.')
from src.dataloader import MovedObjectDataset
from src.utils import load_split_files
from src.config import Config

class DETRTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Load model
        print("Loading DETR model...")
        self.model = DetrForObjectDetection.from_pretrained(config.MODEL_NAME)
        self.model.to(self.device)
        
        # Setup datasets and dataloaders
        self.setup_data()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        print(f"Model loaded. Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_data(self):
        """Setup train and test datasets and dataloaders"""
        print("Setting up datasets...")
        
        # Load train/test split
        train_files, test_files = load_split_files('data')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((self.config.DETR_INPUT_SIZE, self.config.DETR_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        self.train_dataset = MovedObjectDataset(
            annotation_dir=self.config.ANNOTATION_DIR,
            image_base_dir=self.config.DATA_BASE_DIR,
            file_list=train_files,
            transform=transform
        )
        
        self.test_dataset = MovedObjectDataset(
            annotation_dir=self.config.ANNOTATION_DIR,
            image_base_dir=self.config.DATA_BASE_DIR,
            file_list=test_files,
            transform=transform
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows/WSL compatibility
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Test dataset: {len(self.test_dataset)} samples")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, (img1, img2, targets) in enumerate(pbar):
            # Move to device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            # TODO: Implement forward hook and feature extraction here
            # For now, this is a placeholder
            
            # Forward pass (placeholder - will be updated with feature difference)
            # outputs = self.model(pixel_values=img1, labels=targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            # loss = outputs.loss
            # loss.backward()
            # self.optimizer.step()
            
            # total_loss += loss.item()
            # pbar.set_postfix({'loss': loss.item()})
            
            print("TODO: Implement forward pass with feature difference")
            break  # Remove this once implemented
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate on test set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for img1, img2, targets in tqdm(self.test_loader, desc="Validating"):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                # TODO: Implement validation
                # outputs = self.model(pixel_values=img1, labels=targets)
                # total_loss += outputs.loss.item()
                
                print("TODO: Implement validation")
                break  # Remove this once implemented
        
        return total_loss / len(self.test_loader)
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config.NUM_EPOCHS} epochs...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR, 
                    f"checkpoint_epoch_{epoch+1}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    config = Config()
    trainer = DETRTrainer(config)
    trainer.train()