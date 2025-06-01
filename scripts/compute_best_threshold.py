#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import yaml

# ————————————————————————————
# Adjust these imports/paths as needed:
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)

from src.config import load_config
from src.dataset import get_dataloaders
from src.model import get_unet_model
# ————————————————————————————

def compute_threshold_on_validation():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load your best‐model checkpoint
    model = get_unet_model(cfg).to(device)
    ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Get train/val/test loaders; we only need val loader here:
    _, val_loader, _ = get_dataloaders()

    # Collect all y_true and y_prob from the entire val set
    all_true = []
    all_prob = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting validation probabilities"):
            images = batch['image'].to(device)  # shape = [B, 1 or 3, H, W]
            labels = batch['label'].to(device)  # shape = [B, 1, H, W]
            
            # Make sure images have dimensions that are multiples of 16 (common UNet requirement)
            h, w = images.shape[2], images.shape[3]
            new_h, new_w = ((h + 15) // 16) * 16, ((w + 15) // 16) * 16
            
            if h != new_h or w != new_w:
                # Resize or pad images to the required dimensions
                padded_images = torch.zeros(images.shape[0], images.shape[1], new_h, new_w, device=device)
                padded_images[:, :, :h, :w] = images
                images = padded_images
                
                # Also resize labels for consistency when calculating metrics
                padded_labels = torch.zeros(labels.shape[0], labels.shape[1], new_h, new_w, device=device)
                padded_labels[:, :, :h, :w] = labels
                labels = padded_labels

            # raw network output (logits or pre‐sigmoid)
            logits = model(images)
            
            # If we padded, we need to crop back to original size
            if h != new_h or w != new_w:
                logits = logits[:, :, :h, :w]
            
            # Get the original label dimensions before any padding
            orig_h, orig_w = labels.shape[2], labels.shape[3]
            
            # Ensure logits match the shape of original labels
            if logits.shape[2] != orig_h or logits.shape[3] != orig_w:
                logits = torch.nn.functional.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                
            probs = torch.sigmoid(logits)  # shape = [B, 1, H, W]
            # flatten both arrays - ensure both arrays have the same shape
            labels_flat = labels.cpu().view(-1).numpy()
            probs_flat = probs.cpu().view(-1).numpy()
            
            # Verify they have the same shape
            assert labels_flat.shape == probs_flat.shape, f"Shape mismatch: {labels_flat.shape} vs {probs_flat.shape}"
            
            all_true.append(labels_flat)
            all_prob.append(probs_flat)

    all_true = np.concatenate(all_true, axis=0)   # shape = [N_pixels_val]
    all_prob = np.concatenate(all_prob, axis=0)   # shape = [N_pixels_val]

    # Only keep non‐background pixels? Actually, for vessel segmentation we want both
    # background (0) and vessel (1) to tune threshold.
    # Compute Precision‐Recall curve:
    prec, recall, thresholds = precision_recall_curve(all_true, all_prob)
    # F1 = 2 * (P * R) / (P + R)
    f1_scores = 2 * (prec * recall) / (prec + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"[threshold] Best threshold: {best_thresh:.4f}, F1 (Dice) on val: {best_f1:.4f}")

    # Optionally, write this back to config.yaml under a new key:
    cfg_path = os.path.join(project_root, 'configs', 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict['threshold'] = float(best_thresh)
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)

    print(f"[threshold] Saved best threshold={best_thresh:.4f} into {cfg_path} under key 'threshold'.")


if __name__ == '__main__':
    compute_threshold_on_validation()
