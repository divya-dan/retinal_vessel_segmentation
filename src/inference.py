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
from src.model import get_unet_model
from monai.data import Dataset
from torch.utils.data import DataLoader

def infer_single_image(model, image_path, transform, device, cfg):
    sample = {'image': image_path, 'label': image_path}  # Dummy label
    sample = transform(sample)
    image_tensor = sample['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = (output.sigmoid() > 0.5).float()

    visualize_inference(image_tensor[0], pred[0], os.path.basename(image_path), cfg, title="Predicted Vessel Mask")

def infer_batch(model, data_root, transform, device, cfg, num_samples=5):
    test_pairs = get_image_mask_pairs(os.path.join(data_root, 'test'))
    random.shuffle(test_pairs)
    selected = test_pairs[:num_samples]
    dataset = Dataset(data=selected, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(os.path.join(cfg['paths']['output_dir'], 'inference'), exist_ok=True)
    save_path = os.path.join(cfg['paths']['output_dir'], 'inference', 'batch_inference.png')

    fig, axs = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axs = [axs]  # Ensure it's iterable

    for i, (batch, ax_pair) in enumerate(zip(tqdm(loader, desc="Batch inference"), axs)):
        image_tensor = batch['image'].to(device)
        with torch.no_grad():
            output = model(image_tensor)
            pred = (output.sigmoid() > 0.5).float()

        image_np = image_tensor[0].cpu().squeeze().numpy().transpose(1, 2, 0)
        pred_np = pred[0].cpu().squeeze().numpy()

        ax_pair[0].imshow(image_np)
        ax_pair[0].set_title(f"Fundus Image [{i}]")
        ax_pair[1].imshow(pred_np, cmap='gray')
        ax_pair[1].set_title(f"Predicted Vessel Mask [{i}]")
        for ax in ax_pair:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_inference(image_tensor, pred_tensor, filename, cfg, title="Predicted Mask"):
    image_np = image_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
    pred_np = pred_tensor.cpu().squeeze().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(image_np)
    axs[0].set_title("Fundus Image")
    axs[1].imshow(pred_np, cmap='gray')
    axs[1].set_title(title)
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.join(cfg['paths']['output_dir'], 'inference'), exist_ok=True)
    save_path = os.path.join(cfg['paths']['output_dir'], 'inference', filename)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run inference on fundus images.")
    parser.add_argument('--image', type=str, help="Path to a single fundus image.")
    parser.add_argument('--batch', type=int, help="Number of random test samples to run inference on.")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_unet_model(cfg).to(device)
    ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], 'best_model.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    _, _, test_transforms = get_transforms(cfg['data'].get('image_size', (512, 512)))

    if args.image:
        print(f"[infer] Running inference on image: {args.image}")
        infer_single_image(model, args.image, test_transforms, device, cfg)
    elif args.batch:
        print(f"[infer] Running inference on {args.batch} random test images")
        infer_batch(model, cfg['data']['data_root'], test_transforms, device, cfg, num_samples=args.batch)
    else:
        print("[infer] Please provide either --image or --batch option.")

if __name__ == '__main__':
    main()
