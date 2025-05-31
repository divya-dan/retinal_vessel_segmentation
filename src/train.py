#!/usr/bin/env python
import os
import sys

import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader
from monai.utils import set_determinism
from monai.losses import DiceCELoss
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
    criterion = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['scheduler']['step_size'], gamma=cfg['scheduler']['gamma'])

    # Metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Logging
    writer = SummaryWriter(cfg['paths']['log_dir'])
    best_dice = 0.0
    patience = cfg['early_stopping']['patience']
    counter = 0

    for epoch in range(cfg['train']['num_epochs']):
        print(f"Epoch {epoch+1}/{cfg['train']['num_epochs']}")
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar("train/loss", avg_epoch_loss, epoch)

        # Validation
        if (epoch + 1) % cfg['train']['val_interval'] == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
                    val_outputs = model(val_inputs)
                    val_outputs = [AsDiscrete(threshold=0.5)(i) for i in decollate_batch(val_outputs)]
                    val_labels = decollate_batch(val_labels)
                    dice_metric(y_pred=val_outputs, y=val_labels)

            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            writer.add_scalar("val/dice", val_dice, epoch)
            print(f"Validation Dice: {val_dice:.4f}")

            # Checkpointing
            if val_dice > best_dice:
                best_dice = val_dice
                counter = 0
                ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model.pth')
                torch.save(model.state_dict(), ckpt_path)
                print(f"[checkpoint] Saved best model with Dice {val_dice:.4f}")
            else:
                counter += 1

            if counter >= patience:
                print(f"[early stopping] No improvement for {patience} epochs. Stopping early.")
                break

        scheduler.step()

    writer.close()
    print("[train] Training complete.")


if __name__ == '__main__':
    train()
