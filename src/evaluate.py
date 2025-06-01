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
from src.model import get_unet_model
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from skimage.morphology import remove_small_objects, remove_small_holes, closing, square
import numpy as np

def clean_mask(pred_mask: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    - pred_mask:  any array that (after squeezing) becomes shape (H, W).
      (Examples: (1, H, W), (H, W), (B, 1, H, W) with B=1—all of these squeeze to (H, W).)
    - min_size:   the minimum connected‐component size to keep.

    Returns a cleaned 2D uint8 mask of shape (H, W).
    """
    # If it's a PyTorch tensor, convert to numpy:
    if not isinstance(pred_mask, np.ndarray):
        try:
            pred_mask = pred_mask.cpu().numpy()
        except Exception:
            raise ValueError("clean_mask() expects a numpy array or torch tensor convertible to numpy")

    # Squeeze out any singleton dims until we have exactly 2 dims:
    squeezed = np.squeeze(pred_mask)
    if squeezed.ndim != 2:
        raise ValueError(f"clean_mask() requires a mask that squeezes to 2D; got shape {pred_mask.shape} → squeezed to {squeezed.shape}")

    mask2d = squeezed.astype(bool)  # make sure it's boolean for the morphology ops

    # 1) Remove components smaller than min_size
    cleaned = remove_small_objects(mask2d, min_size=min_size)
    # 2) Fill holes smaller than min_size
    cleaned = remove_small_holes(cleaned, area_threshold=min_size)
    # 3) Apply a 3×3 closing to smooth out tiny notches
    cleaned = closing(cleaned, square(3))

    return cleaned.astype(np.uint8)

def evaluate():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataloader
    _, _, test_loader = get_dataloaders()

    # Load model and weights
    model = get_unet_model(cfg).to(device)
    ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model.pth')
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
            
            # Apply clean_mask to each prediction in the batch
            # cleaned_preds = []
            # for pred in preds:
            #     pred_np = pred.cpu().squeeze().numpy()
            #     cleaned_pred_np = clean_mask(pred_np)
            #     cleaned_pred = torch.from_numpy(cleaned_pred_np).unsqueeze(0).unsqueeze(0).to(device).float()
            #     cleaned_preds.append(cleaned_pred)
            
            # preds = torch.cat(cleaned_preds, dim=0)

            dice = dice_metric(preds, labels)
            all_dice.append(dice.cpu().numpy())  # Store the tensor as numpy array

            if i < 5:  # Save visualization of first 5 samples
                visualize_sample(images[0], labels[0], preds[0], i, cfg)

    # Flatten the list of arrays if needed and calculate mean
    all_dice = np.concatenate(all_dice, axis=0) if len(all_dice) > 0 else np.array([])
    mean_dice = np.mean(all_dice)
    print(f"[eval] Mean Dice Score: {mean_dice:.4f}")


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


if __name__ == '__main__':
    evaluate()
