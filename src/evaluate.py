#!/usr/bin/env python
import os
import sys
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add project root to PYTHONPATH
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)

from src.config import load_config
from src.dataset import get_dataloaders
from src.model import get_model
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from skimage.morphology import remove_small_objects, remove_small_holes, closing, square
import numpy as np
import argparse


def visualize_sample(image_tensor, label_tensor, pred_tensor, index, cfg):
    """
    Save side-by-side image of input, label, prediction.
    """
    image_np = image_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
    label_np = label_tensor.cpu().squeeze().numpy()
    pred_np = pred_tensor.cpu().squeeze().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_np)
    axs[0].set_title("Fundus Image")
    axs[1].imshow(label_np, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred_np, cmap='gray')
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(cfg['paths']['output_dir'], 'eval_samples', f'sample_{index}.png')
    plt.savefig(save_path)
    plt.close()

def evaluate(config_path=None):
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = load_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataloader
    _, _, test_loader = get_dataloaders()

    # Load model and weights
    model = get_model(cfg).to(device)
    ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model_dice.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    print("[eval] Running inference on test set...")
    all_dice = []
    os.makedirs(os.path.join(cfg['paths']['output_dir'], 'eval_samples'), exist_ok=True)

    # Ensure roi_size is a list of integers
    roi_size_config = cfg.get("sliding_window", {}).get("roi_size", [512, 512])
    roi_size = [int(x) if isinstance(x, str) else x for x in roi_size_config] if isinstance(roi_size_config, list) else [512, 512]
    
    # Convert other parameters to appropriate types
    sw_batch_size = int(cfg.get("sliding_window", {}).get("sw_batch_size", 4))
    overlap = float(cfg.get("sliding_window", {}).get("overlap", 0.25))

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # ─── Run sliding‐window on raw logits, using Gaussian blending ───
            outputs = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian"           
            )
            # ─── Only now convert the entire stitched logits to probabilities and threshold ───
            threshold = float(cfg.get("threshold", 0.55))
            probs = torch.sigmoid(outputs)    
            preds = (probs > threshold).float()

            dice = dice_metric(preds, labels)
            all_dice.append(dice.cpu().numpy())  # Store the tensor as numpy array

            if i < 5:  # Save visualization of first 5 samples
                visualize_sample(images[0], labels[0], preds[0], i, cfg)

    # Flatten the list of arrays if needed and calculate mean
    all_dice = np.concatenate(all_dice, axis=0) if len(all_dice) > 0 else np.array([])
    mean_dice = np.mean(all_dice)
    print(f"[eval] Mean Dice Score: {mean_dice:.4f}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Evaluate vessel segmentation model")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    evaluate(args.config)
