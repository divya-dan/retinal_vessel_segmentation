#!/usr/bin/env python
import os
import sys
import glob
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to PYTHONPATH so modules in src/ can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, project_root)

from sklearn.model_selection import train_test_split
from src.config import load_config

from monai import transforms as mt
from monai.transforms import Compose, RandCropByPosNegLabeld

def get_image_mask_pairs(folder: str):
    fundus_dir = os.path.join(folder, 'fundus')
    vessels_dir = os.path.join(folder, 'vessels')
    img_patterns = ['*.png', '*.jpg', '*.jpeg', '*.ppm', '*.tif']
    fundus_files = []
    for pat in img_patterns:
        fundus_files.extend(glob.glob(os.path.join(fundus_dir, pat)))
    fundus_files = sorted(fundus_files)

    pairs = []
    for img_path in fundus_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_found = False
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            mask_path = os.path.join(vessels_dir, base_name + ext)
            if os.path.exists(mask_path):
                pairs.append({'image': img_path, 'label': mask_path})
                mask_found = True
                break
        if not mask_found:
            print(f"[preprocess] Warning: No matching mask for {img_path}")
    return pairs

def split_train_val(train_folder: str, split_ratio: float, seed: int):
    all_pairs = get_image_mask_pairs(train_folder)
    if len(all_pairs) == 0:
        raise ValueError("No image-mask pairs found in 'vessels/'.")
    random.seed(seed)
    train_pairs, val_pairs = train_test_split(
        all_pairs, test_size=split_ratio, random_state=seed
    )
    print(f"[preprocess] Total: {len(all_pairs)}, Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    return train_pairs, val_pairs

def get_transforms(image_size=(1024, 1024), use_patches=False, patch_size=(256,256),
                   pos=1, neg=1, num_samples=4):
    if use_patches:
        train_transforms = Compose([
            mt.LoadImaged(keys=['image', 'label']),
            mt.EnsureChannelFirstd(keys=['image', 'label']),
            mt.ScaleIntensityd(keys=['image']),
            RandCropByPosNegLabeld(
                keys=['image','label'], label_key='label',
                spatial_size=patch_size, pos=pos, neg=neg, num_samples=num_samples,
                image_key='image', image_threshold=0
            ),
            mt.RandFlipd(keys=['image','label'], prob=0.5, spatial_axis=0),
            mt.RandRotate90d(keys=['image','label'], prob=0.5, max_k=3),
            mt.RandGaussianNoised(keys='image', prob=0.3, mean=0.0, std=0.1),
            mt.RandBiasFieldd(keys='image', prob=0.3, coeff_range=(0.15, 0.5)),
            mt.RandAdjustContrastd(keys='image', prob=0.3, gamma=(0.7, 1.5)),
            mt.RandZoomd(keys=['image','label'], prob=0.3, min_zoom=0.9, max_zoom=1.1),
            mt.ToTensord(keys=['image','label']),
        ])
    else:
        train_transforms = Compose([
            mt.LoadImaged(keys=['image','label']),
            mt.EnsureChannelFirstd(keys=['image','label']),
            mt.ScaleIntensityd(keys=['image']),
            mt.RandFlipd(keys=['image','label'], prob=0.5, spatial_axis=0),
            mt.RandRotate90d(keys=['image','label'], prob=0.5, max_k=3),
            mt.RandGaussianNoised(keys='image', prob=0.3, mean=0.0, std=0.1),
            mt.RandBiasFieldd(keys='image', prob=0.3, coeff_range=(0.15, 0.5)),
            mt.RandAdjustContrastd(keys='image', prob=0.3, gamma=(0.7, 1.5)),
            mt.RandZoomd(keys=['image','label'], prob=0.3, min_zoom=0.9, max_zoom=1.1),
            mt.Resized(keys=['image','label'], spatial_size=image_size, mode=['bilinear','nearest']),
            mt.ToTensord(keys=['image','label']),
        ])

    val_transforms = Compose([
        mt.LoadImaged(keys=['image','label']),
        mt.EnsureChannelFirstd(keys=['image','label']),
        mt.ScaleIntensityd(keys=['image']),
        mt.Resized(keys=['image','label'], spatial_size=image_size, mode=['bilinear','nearest']),
        mt.ToTensord(keys=['image','label']),
    ])
    test_transforms = val_transforms

    return train_transforms, val_transforms, test_transforms

def visualize_augmentations(sample_pair, train_transforms):
    steps = train_transforms.transforms
    init_steps = steps[:3]
    tail_steps = [t for t in steps[3:] if not isinstance(t, (mt.Resized, mt.ToTensord))]

    # load once
    sample = {'image': sample_pair['image'], 'label': sample_pair['label']}
    for t in init_steps:
        sample = t(sample)
        # if we get back a list (from RandCropByPosNegLabeld), grab the first crop
        if isinstance(sample, list):
            sample = sample[0]

    imgs = [('Original', (sample['image'], sample['label']))]

    for t in tail_steps:
        sample = t(sample)
        # again, unwrap lists
        if isinstance(sample, list):
            sample = sample[0]

        imgs.append((t.__class__.__name__, (sample['image'], sample['label'])))

    # plotting code unchanged…
    n = len(imgs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()
    for i, (name, (img, lbl)) in enumerate(imgs):
        arr = img.numpy() if hasattr(img, 'numpy') else img
        if arr.ndim == 3:
            arr = np.transpose(arr, (1,2,0))
            if arr.shape[2] == 1:
                arr = arr[...,0]
        axes[i].imshow(arr, cmap='gray' if arr.ndim==2 else None)
        m = lbl.numpy() if hasattr(lbl, 'numpy') else lbl
        if m.ndim > 2:
            m = m.squeeze()
        axes[i].imshow(m, cmap='jet', alpha=0.3)
        axes[i].set_title(name)
        axes[i].axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig('augmentation_visualization.png')
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle
from monai import transforms as mt
from monai.transforms import Compose, RandCropByPosNegLabeld
from matplotlib.gridspec import GridSpec

def visualize_augmentations_with_box(sample_pair, train_transforms, patch_size):
    # --- load full image & label as before ---
    loader = Compose([
        mt.LoadImaged(keys=['image','label']),
        mt.EnsureChannelFirstd(keys=['image','label']),
        mt.ScaleIntensityd(keys=['image'])
    ])
    data = loader({'image': sample_pair['image'], 'label': sample_pair['label']})
    img_tensor, lbl_tensor = data['image'], data['label']   # (1,H,W)
    img_disp = img_tensor[0].numpy()
    H, W = img_disp.shape
    ph, pw = patch_size

    # pick a random patch
    y0 = random.randint(0, H - ph)
    x0 = random.randint(0, W - pw)

    # figure out how many augmentations we’ll show
    steps = train_transforms.transforms
    tail_steps = [
        t for t in steps
        if not isinstance(t, (mt.LoadImaged, mt.EnsureChannelFirstd,
                               mt.ScaleIntensityd, RandCropByPosNegLabeld,
                               mt.Resized, mt.ToTensord))
    ]
    n_cols = 1 + len(tail_steps)  # 1 for original patch, rest for each aug

    # --- Set up GridSpec: 2 rows, n_cols columns,
    # row 0 spans all cols, row 1 has n_cols small subplots ---
    fig = plt.figure(figsize=(4*n_cols, 8))
    gs = GridSpec(2, n_cols, height_ratios=[1,1], figure=fig)

    # Row 0: full image + red box, spanning all columns
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.imshow(img_disp, cmap='gray')
    rect = Rectangle((x0, y0), pw, ph, edgecolor='red', linewidth=2, fill=False)
    ax_full.add_patch(rect)
    ax_full.set_title('Full image with patch box')
    ax_full.axis('off')

    # Crop out the patch
    patch_img = img_tensor[:, y0:y0+ph, x0:x0+pw]
    patch_lbl = lbl_tensor[:, y0:y0+ph, x0:x0+pw]

    # row 1, col 0: original patch
    ax0 = fig.add_subplot(gs[1, 0])
    orig = patch_img[0].numpy()
    ax0.imshow(orig, cmap='gray')
    ax0.imshow(patch_lbl[0].numpy(), cmap='jet', alpha=0.3)
    ax0.set_title('Original patch')
    ax0.axis('off')

    # now run and plot each augmentation in its own column
    sample = {'image': patch_img, 'label': patch_lbl}
    for i, t in enumerate(tail_steps, start=1):
        out = t(sample)
        if isinstance(out, list): out = out[0]
        sample = out

        img_a = sample['image'].numpy()   # (1,ph,pw)
        lbl_a = sample['label'].numpy()
        disp_a = img_a[0]  # just H×W now

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(disp_a, cmap='gray')
        ax.imshow(lbl_a[0], cmap='jet', alpha=0.3)
        ax.set_title(t.__class__.__name__)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_visualization.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess fundus images")
    parser.add_argument('--augment', action='store_true',
                        help="Visualize each augmentation step on one sample")
    args = parser.parse_args()

    cfg = load_config()
    train_folder = os.path.join(cfg['data']['data_root'], 'train')
    split_ratio = cfg['data']['train_val_split']
    seed = cfg['data']['random_seed']
    train_pairs, val_pairs = split_train_val(train_folder, split_ratio, seed)

    patch_cfg = cfg.get('patch', {})
    train_t, val_t, test_t = get_transforms(
        image_size=tuple(cfg['data'].get('image_size', (1024,1024))),
        use_patches=patch_cfg.get('use', False),
        patch_size=tuple(patch_cfg.get('size', (256,256))),
        pos=patch_cfg.get('pos', 1),
        neg=patch_cfg.get('neg', 1),
        num_samples=patch_cfg.get('num_samples', 4)
    )

    if args.augment:
        # unpack patch size and call the new visualization
        ph, pw = patch_cfg.get('size', (256, 256))
        visualize_augmentations_with_box(
            sample_pair=train_pairs[0],
            train_transforms=train_t,
            patch_size=(ph, pw)
        )
        sys.exit(0)

        print("[preprocess] Augmented transforms ready.")
