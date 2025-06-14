#!/usr/bin/env python
import os
import sys

import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader
from monai.utils import set_determinism
from monai.losses import DiceCELoss, DiceFocalLoss

from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data.utils import decollate_batch
from tqdm import tqdm

# Add project root to PYTHONPATH for imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
# Actually project root is one level up from src/
project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

from src.config import load_config
from src.dataset import get_transforms, split_train_val
from src.model import get_unet_model

from torch.utils.data import Dataset

class SimpleSegmentationDataset(Dataset):
    def __init__(self, data_list, transforms):
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.transforms(self.data_list[idx])

set_determinism(seed=42)

def train():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_pairs, val_pairs = split_train_val(
        os.path.join(cfg['data']['data_root'], 'train'),
        cfg['data']['train_val_split'],
        cfg['data']['random_seed']
    )
    train_t, val_t, _ = get_transforms()

    # Create dataset instance
    train_ds = SimpleSegmentationDataset(train_pairs, train_t)

    val_ds = SimpleSegmentationDataset(val_pairs, val_t)

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'])

    # Build model
    model = get_unet_model(cfg).to(device)
    if cfg.get('train', {}).get('multi_gpu', False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Loss, optimizer, scheduler
    # Choose loss function based on config
    if cfg.get('train', {}).get('loss_function', 'dice_ce') == 'dice_focal':
        criterion = DiceFocalLoss(sigmoid=True)
    else:
        criterion = DiceCELoss(sigmoid=True)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg['train']['learning_rate'],
        weight_decay=cfg['train'].get('weight_decay', 1e-5)  # Add L2 regularization
    )
    
    # Improved scheduler - Cosine Annealing with Warm Restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=cfg['scheduler'].get('T_0', 10),
        T_mult=cfg['scheduler'].get('T_mult', 2),
        eta_min=cfg['scheduler'].get('eta_min', 1e-6)
    )

    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Logging
    writer = SummaryWriter(cfg['paths']['log_dir'])
    
    # Track both validation loss and dice
    best_val_loss = float('inf')
    best_dice = 0.0
    patience = cfg['early_stopping']['patience']
    counter = 0
    
    # Learning rate tracking
    initial_lr = cfg['train']['learning_rate']

    for epoch in range(cfg['train']['num_epochs']):
        print(f"Epoch {epoch+1}/{cfg['train']['num_epochs']}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], epoch)

        # Validation phase
        if (epoch + 1) % cfg['train']['val_interval'] == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            dice_metric.reset()
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
                    val_outputs = model(val_inputs)
                    
                    # Calculate validation loss
                    v_loss = criterion(val_outputs, val_labels)
                    val_loss += v_loss.item()
                    val_batches += 1
                    
                    # Calculate Dice metric
                    val_outputs_discrete = [AsDiscrete(threshold=0.5)(i) for i in decollate_batch(val_outputs)]
                    val_labels_discrete = decollate_batch(val_labels)
                    dice_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)

            avg_val_loss = val_loss / val_batches
            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            
            # Log validation metrics
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            writer.add_scalar("val/dice", val_dice, epoch)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {val_dice:.4f}")

            # Model selection based on validation loss (lower is better)
            improved = False
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                improved = True
                counter = 0
                
                # Save best model based on validation loss
                ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model_val_loss.pth')
                if hasattr(model, 'module'):  # DataParallel case
                    torch.save(model.module.state_dict(), ckpt_path)
                else:
                    torch.save(model.state_dict(), ckpt_path)
                print(f"[checkpoint] Saved best model with Val Loss {avg_val_loss:.4f}")
            
            # Also save best Dice model for comparison
            if val_dice > best_dice:
                best_dice = val_dice
                ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model_dice.pth')
                if hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), ckpt_path)
                else:
                    torch.save(model.state_dict(), ckpt_path)
                print(f"[checkpoint] Saved best Dice model with score {val_dice:.4f}")
            
            if not improved:
                counter += 1

            if counter >= patience:
                print(f"[early stopping] No improvement in validation loss for {patience} epochs. Stopping early.")
                break

        scheduler.step()
        
        # Periodic checkpoint saving
        if (epoch + 1) % cfg['train'].get('checkpoint_interval', 50) == 0:
            ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), ckpt_path)
            else:
                torch.save(model.state_dict(), ckpt_path)

    writer.close()
    print("[train] Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == '__main__':
    train()
