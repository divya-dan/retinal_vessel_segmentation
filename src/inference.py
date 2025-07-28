#!/usr/bin/env python
import os
import sys
import torch
import argparse
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add project root to PYTHONPATH
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)

from src.config import load_config
from src.dataset import get_image_mask_pairs, get_transforms
from src.model import get_model
from monai.data import Dataset
from torch.utils.data import DataLoader


def calculate_dice_score(pred, target, smooth=1e-8):
    """Calculate Dice coefficient between prediction and target masks."""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def infer_single_image(model, image_path, transform, device, cfg):
    # single-image inference (no ground truth available here)
    sample = {'image': image_path, 'label': image_path}  # Dummy label placeholder
    sample = transform(sample)
    image_tensor = sample['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        threshold = float(cfg.get("threshold", 0.55))
        pred = (output.sigmoid() > threshold).float()

    print(f"[INFO] Single image inference completed for: {os.path.basename(image_path)}")
    print(f"[INFO] No dice score available (ground truth not provided)")

    visualize_inference(
        image_tensor[0],
        pred[0],
        os.path.basename(image_path),
        cfg,
        title="Predicted Vessel Mask"
    )


def infer_batch(model, data_root, transform, device, cfg, num_samples=5):
    # batch inference: show fundus, ground truth, and prediction side by side
    # ensure reproducible selection
    seed = int(cfg.get("inference_seed", 19))
    random.seed(seed)
    np.random.seed(seed)

    test_pairs = get_image_mask_pairs(os.path.join(data_root, 'test'))
    random.shuffle(test_pairs)
    selected = test_pairs[:num_samples]
    dataset = Dataset(data=selected, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    out_dir = os.path.join(cfg['paths']['output_dir'], 'inference')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'batch_inference.png')

    # change to 3 columns: Input, Ground Truth, Prediction
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples), dpi=300)
    if num_samples == 1:
        axs = [axs]

    dice_scores = []
    print(f"\n{'='*60}")
    print(f"{'BATCH INFERENCE RESULTS':^60}")
    print(f"{'='*60}")
    print(f"{'Sample':<8} {'Dice Score':<12} {'Threshold':<12}")
    print(f"{'-'*60}")

    for i, (batch, ax_row) in enumerate(zip(tqdm(loader, desc="Batch inference"), axs)):
        image_tensor = batch['image'].to(device)
        label_tensor = batch['label'].to(device)

        with torch.no_grad():
            output = model(image_tensor)
            threshold = float(cfg.get("threshold", 0.55))
            pred = (output.sigmoid() > threshold).float()

        # Calculate dice score
        dice_score = calculate_dice_score(pred[0], label_tensor[0])
        dice_scores.append(dice_score)
        
        # Print individual score
        print(f"{i+1:<8} {dice_score:<12.4f} {threshold:<12.2f}")

        # convert to numpy arrays
        image_np = image_tensor[0].cpu().squeeze().numpy().transpose(1, 2, 0)
        label_np = label_tensor[0].cpu().squeeze().numpy()
        pred_np = pred[0].cpu().squeeze().numpy()

        # plot side-by-side with dice score in title
        ax_row[0].imshow(image_np)
        ax_row[0].set_title(f"Fundus Image [{i+1}]")
        ax_row[1].imshow(label_np, cmap='gray')
        ax_row[1].set_title(f"Ground Truth [{i+1}]")
        ax_row[2].imshow(pred_np, cmap='gray')
        ax_row[2].set_title(f"Prediction [{i+1}]\nDice: {dice_score:.4f}")

        for ax in ax_row:
            ax.axis('off')

    # Print summary statistics
    print(f"{'-'*60}")
    print(f"{'SUMMARY STATISTICS':^60}")
    print(f"{'-'*60}")
    print(f"Mean Dice Score:    {np.mean(dice_scores):.4f}")
    print(f"Std Dice Score:     {np.std(dice_scores):.4f}")
    print(f"Min Dice Score:     {np.min(dice_scores):.4f}")
    print(f"Max Dice Score:     {np.max(dice_scores):.4f}")
    print(f"{'='*60}\n")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return dice_scores


def visualize_inference(image_tensor, pred_tensor, filename, cfg, title="Predicted Mask"):
    # single-image plotting (no ground truth)
    image_np = image_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
    pred_np = pred_tensor.cpu().squeeze().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    axs[0].imshow(image_np)
    axs[0].set_title("Fundus Image")
    axs[1].imshow(pred_np, cmap='gray')
    axs[1].set_title(title)
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()

    out_dir = os.path.join(cfg['paths']['output_dir'], 'inference')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run inference on fundus images.")
    parser.add_argument('--image', type=str, help="Path to a single fundus image.")
    parser.add_argument('--batch', type=int, help="Number of random test samples to run inference on.")
    parser.add_argument('--config', type=str, default=None, help="Path to the configuration file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # set random seeds for reproducible batch sampling
    seed = int(cfg.get("inference_seed", 19))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(cfg).to(device)
    ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model_dice.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    _, _, test_transforms = get_transforms(cfg['data'].get('image_size', (512, 512)))

    if args.image:
        print(f"[infer] Running inference on image: {args.image}")
        infer_single_image(model, args.image, test_transforms, device, cfg)
    elif args.batch:
        print(f"[infer] Running inference on {args.batch} random test images")
        dice_scores = infer_batch(model, cfg['data']['data_root'], test_transforms, device, cfg, num_samples=args.batch)
    else:
        print("[infer] Please provide either --image or --batch option.")

if __name__ == '__main__':
    main()