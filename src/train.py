import torch
from torch.utils.data import DataLoader

from src.model import MovedObjectDETR
from src.dataloader import MovedObjectDataset
from src.config import Config

import torchvision.transforms as T
import os

#1. COLLATE FUNCTION
def collate_fn(batch):
    
    #Separate into lists
    img1_list = [sample[0] for sample in batch]  # [img1_0, img1_1, img1_2]
    img2_list = [sample[1] for sample in batch]  # [img2_0, img2_1, img2_2]
    target_list = [sample[2] for sample in batch]  # [target_0, target_1, target_2]
    
    #Stack images (they're all same size)
    img1_batch = torch.stack(img1_list)  
    img2_batch = torch.stack(img2_list)  

    
    return img1_batch, img2_batch, target_list


#2. FINE-TUNING SETUP
def setup_finetuning(model, strategy='all'):
    """
    Configure which parameters to train based on strategy.
    
    Strategies:
        'all'          - Train all parameters
        'head_only'    - Only train classification + bbox heads
        'transformer'  - Train transformer + heads, freeze backbone
        'custom_layer' - Train our custom layer4 + heads
    """
    
    if strategy == 'all':
        # Unfreeze everything
        for name, param in model.named_parameters():
            param.requires_grad = True

    else:
        # Freeze everything first
        for name, param in model.named_parameters():
            param.requires_grad = False

        if strategy == "head_only":
            for name, param in model.named_parameters():
                if 'class_labels_classifier' in name or 'bbox_predictor' in name:
                    param.requires_grad = True

        elif strategy == "transformer":
            for name, param in model.named_parameters():
                if 'backbone' not in name:
                    param.requires_grad = True

        elif strategy == 'custom_layer':
            for name, param in model.named_parameters():
                if 'layer4' in name or 'class_labels_classifier' in name or 'bbox_predictor' in name:
                    param.requires_grad = True

    # Return only trainable parameters
    return [p for p in model.parameters() if p.requires_grad]


    
    
    
    

    
    





#3. OPTIMIZER CREATION
def create_optimizer(model, base_lr=1e-4, weight_decay=1e-4):
    # Get only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=base_lr,
        weight_decay=weight_decay
    )
    
    return optimizer


# 4. TRAINING LOOP
def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()  # Set to training mode (enables dropout, etc.)
    total_loss = 0
    
    for batch_idx, (img1, img2, targets) in enumerate(dataloader):
        # Step 1: Move to device
        img1 = img1.to(device)
        img2 = img2.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Step 2: Forward pass
        outputs = model(img1, img2, targets)
        loss = outputs.loss
        
        # Step 3: Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Step 4: Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        # Step 5: Update weights
        optimizer.step()
        
        # Step 6: Track loss
        total_loss += loss.item()
        
        #progress
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

# 5. CHECKPOINTING
def save_checkpoint(model, optimizer, epoch, loss, strategy, path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'strategy': strategy,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer=None):
    #load training checkpoint
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return checkpoint['epoch']


# MAIN
def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Get hyperparameters from Config
    num_epochs = Config.NUM_EPOCHS
    batch_size = Config.BATCH_SIZE
    learning_rate = Config.LEARNING_RATE
    strategy = Config.FINETUNING_STRATEGY  # Read from config file!

    # Create transform
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = MovedObjectDataset(
        annotation_dir=Config.ANNOTATION_DIR,
        image_base_dir=Config.DATA_BASE_DIR,
        transform=transform
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Batches per epoch: {len(dataloader)}")


    # Create model
    model = MovedObjectDETR(Config.NUM_CLASSES)
    model.to(device)

    # Setup fine-tuning
    trainable_params = setup_finetuning(model, strategy=strategy)
    print(f"Strategy: {strategy}")
    print(f"Trainable parameters: {len(trainable_params)}")
    

    # Create optimizer
    optimizer = create_optimizer(model, base_lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch}/{num_epochs} complete. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss, strategy,
                f"outputs/checkpoints/checkpoint_epoch_{epoch}.pth"
            )


    # Save final model
    save_checkpoint(
        model, optimizer, num_epochs, avg_loss, strategy,
        f"outputs/checkpoints/final_model_{strategy}.pth"
    )

if __name__ == '__main__':
    main()