#!/usr/bin/env python
import os
import sys
import argparse
import csv
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to PYTHONPATH
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)

from src.config import load_config
from src.model import get_model
from src.data_processing.preprocess import get_image_mask_pairs, get_transforms
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader


def save_side_by_side(fundus, gt, pred, out_path, dice):
    """
    fundus: HxWxC numpy array
    gt:     HxW numpy array
    pred:   HxW numpy array
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(fundus)
    axs[0].set_title("Fundus")
    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title(f"Prediction\nDice: {dice:.4f}")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize test results and generate best/worst visualizations.")
    parser.add_argument('--config', type=str, help="Path to YAML config file", required=True)
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    data_root = cfg['data']['data_root']

    # Build test loader manually
    test_folder = os.path.join(data_root, 'test')
    test_pairs = get_image_mask_pairs(test_folder)
    # get_transforms returns (train_t, val_t, test_t)
    _, _, test_t = get_transforms(tuple(cfg['data']['image_size']))
    test_ds = Dataset(data=test_pairs, transform=test_t)
    num_workers = cfg.get('data', {}).get('n_workers', os.cpu_count())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    # Load model
    device = torch.device(cfg.get('train', {}).get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg).to(device)
    ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model_dice.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Prepare metric and outputs
    dice_metric = DiceMetric(include_background=False, reduction='none')
    threshold = float(cfg.get('threshold', 0.5))
    roi_size = list(cfg['sliding_window']['roi_size'])
    sw_batch = int(cfg['sliding_window']['sw_batch_size'])
    overlap = float(cfg['sliding_window']['overlap'])

    out_metrics = []  # list of dicts: {dice, fundus, gt, pred}
    records = []      # tuple list for visualization

    # Iterate over test set
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            image = batch['image'].to(device)    # shape [1,C,H,W]
            label = batch['label'].to(device)    # shape [1,1,H,W]

            # Sliding-window inference with gaussian blending
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch,
                predictor=model,
                overlap=overlap,
                mode='gaussian'
            )
            prob = torch.sigmoid(logits)
            pred = (prob > threshold).float()

            # Compute Dice per sample
            d = dice_metric(pred, label).cpu().numpy().item()
            # Convert to numpy images
            fundus_np = image.cpu().squeeze().numpy().transpose(1,2,0)
            gt_np     = label.cpu().squeeze().numpy()
            pred_np   = pred.cpu().squeeze().numpy()

            records.append((d, fundus_np, gt_np, pred_np))

    # Compute summary stats
    dice_vals = [r[0] for r in records]
    mean_dice = float(np.mean(dice_vals))
    std_dice  = float(np.std(dice_vals))

    # Prepare output dirs
    metrics_dir = os.path.join(cfg['paths']['output_dir'], 'metrics')
    figs_dir    = os.path.join(cfg['paths']['output_dir'], 'figures', 'qualitative')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # Write summary CSV
    csv_path = os.path.join(metrics_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'mean_dice', 'std_dice'])
        tag = os.path.splitext(os.path.basename(args.config))[0]
        writer.writerow([tag, mean_dice, std_dice])
    print(f"Saved summary to {csv_path}")

    # Best and worst examples
    best = max(records, key=lambda x: x[0])
    worst = min(records, key=lambda x: x[0])
    best_path = os.path.join(figs_dir, f"{tag}_best.png")
    worst_path = os.path.join(figs_dir, f"{tag}_worst.png")
    save_side_by_side(best[1], best[2], best[3], best_path, dice=best[0])
    save_side_by_side(worst[1], worst[2], worst[3], worst_path, dice=worst[0])
    print(f"Saved best/worst figures to {figs_dir}")

    print(f"Metrics: mean_dice={mean_dice:.4f}, std_dice={std_dice:.4f}")

if __name__ == '__main__':
    main()
