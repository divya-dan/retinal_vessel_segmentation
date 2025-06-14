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


from monai import transforms as mt
from monai.transforms import (
    Compose, RandCropByPosNegLabeld, OneOf, Identityd, SomeOf,
)

def get_transforms(
    image_size=(1024, 1024), use_patches=False, patch_size=(256,256),
    pos=1, neg=1, num_samples=4,
):
    # --- 1) Base loading & scaling ---
    base = [
        mt.LoadImaged(keys=['image','label']),
        mt.EnsureChannelFirstd(keys=['image','label']),
        mt.ScaleIntensityd(keys=['image']),
    ]
    if use_patches:
        base.append(RandCropByPosNegLabeld(
            keys=['image','label'], label_key='label',
            spatial_size=patch_size, pos=pos, neg=neg,
            num_samples=num_samples, image_key='image',
            image_threshold=0
        ))

    # --- 2) Light / “standard” augmentations ---
    standard = [
        mt.RandFlipd(keys=['image','label'], prob=0.5, spatial_axis=0),
        mt.RandRotate90d(keys=['image','label'], prob=0.5, max_k=3),
        mt.RandZoomd(keys=['image','label'], prob=0.4,
                     min_zoom=0.9, max_zoom=1.1),
        mt.RandAdjustContrastd(keys='image', prob=0.3,
                               gamma=(0.8,1.2)),
    ]

    # --- 3) Heavy spatial (pick 1 of 4 including skip for every sample) ---
    heavy_spatial = SomeOf(
        transforms=[
            Identityd(keys=['image','label']),     # skip heavy transform (70%)
            mt.Rand2DElasticd(
                keys=['image','label'], prob=1.0,
                spacing=(4, 4), magnitude_range=(2, 2),
                mode=('bilinear','nearest')
            ),
            mt.RandGridDistortiond(
                keys=['image','label'], prob=1.0,
                num_cells=7, distort_limit=0.3
            ),
            mt.RandAffined(
                keys=['image','label'], prob=1.0,
                rotate_range=(0, 0), shear_range=(0.1, 0.1),
                translate_range=(10, 10), scale_range=(0.9, 1.1),
                mode=('bilinear','nearest'), padding_mode='reflection'
            ),
        ],
        num_transforms=1,
        weights=[0.7, 0.1, 0.1, 0.1],  # skip 70%, each heavy 10%
        map_items=True,
    )

    # --- 4) Heavy intensity (pick 1 of 3 including skip for every sample) ---
    heavy_intensity = SomeOf(
        transforms=[
            Identityd(keys=['image']),           # skip heavy op (80%)
            mt.RandRicianNoised(
                keys=['image'], prob=1.0,
                mean=0.0, std=0.05
            ),
            mt.RandHistogramShiftd(
                keys=['image'], prob=1.0,
                num_control_points=256
            ),
        ],
        num_transforms=1,
        weights=[0.8, 0.1, 0.1],  # skip 80%, each heavy 10%
        map_items=True,
    )

    # --- 5) Final resize / to tensor ---
    final = []
    if not use_patches:
        final.append(mt.Resized(
            keys=['image','label'],
            spatial_size=image_size,
            mode=['bilinear','nearest']
        ))
    final.append(mt.ToTensord(keys=['image','label']))

    train_transforms = Compose(
        base + standard + [heavy_spatial] + [heavy_intensity] + final
    )

    val_transforms = Compose([
        mt.LoadImaged(keys=['image','label']),
        mt.EnsureChannelFirstd(keys=['image','label']),
        mt.ScaleIntensityd(keys=['image']),
        mt.Resized(
            keys=['image','label'],
            spatial_size=image_size,
            mode=['bilinear','nearest']
        ),
        mt.ToTensord(keys=['image','label']),
    ])
    test_transforms = val_transforms

    return train_transforms, val_transforms, test_transforms


import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from monai import transforms as mt
from monai.transforms import Compose, RandCropByPosNegLabeld

def visualize_augmentations_grid(sample_pairs, train_transforms, patch_size):
    """
    sample_pairs: list of dicts [{'image':path,'label':path},…] (length=5)
    train_transforms: your Compose() pipeline
    patch_size: (ph, pw)
    """
    # 1) Build a loader for full-image display
    loader = Compose([
        mt.LoadImaged(keys=['image','label']),
        mt.EnsureChannelFirstd(keys=['image','label']),
        mt.ScaleIntensityd(keys=['image'])
    ])
    # 2) Extract only the “tail” augmentations you want to show per patch
    steps = train_transforms.transforms
    tail_steps = [
        t for t in steps
        if not isinstance(t, (
            mt.LoadImaged, mt.EnsureChannelFirstd, mt.ScaleIntensityd,
            RandCropByPosNegLabeld, mt.Resized, mt.ToTensord
        ))
    ]

    n = len(sample_pairs)
    ncols = 2 + len(tail_steps)  # full image, original patch, + each aug
    fig, axes = plt.subplots(n, ncols, figsize=(4*ncols, 4*n))

    for i, sample_pair in enumerate(sample_pairs):
        # — Load full image & label —
        data = loader({'image': sample_pair['image'], 'label': sample_pair['label']})
        img_tensor = data['image']   # (1, H, W)
        lbl_tensor = data['label']   # (1, H, W)
        img_disp = img_tensor[0].numpy()
        H, W = img_disp.shape
        ph, pw = patch_size

        # — Pick a random patch origin and draw box —
        y0 = random.randint(0, H - ph)
        x0 = random.randint(0, W - pw)
        ax_full = axes[i, 0]
        ax_full.imshow(img_disp, cmap='gray')
        rect = Rectangle((x0, y0), pw, ph,
                         edgecolor='red', linewidth=2, fill=False)
        ax_full.add_patch(rect)
        ax_full.set_title('Full image')
        ax_full.axis('off')

        # — Crop and show the original patch —
        patch_img = img_tensor[:, y0:y0+ph, x0:x0+pw]
        patch_lbl = lbl_tensor[:, y0:y0+ph, x0:x0+pw]
        ax_orig = axes[i, 1]
        disp_patch = patch_img[0].numpy()
        ax_orig.imshow(disp_patch, cmap='gray')
        ax_orig.imshow(patch_lbl[0].numpy(), cmap='jet', alpha=0.3)
        ax_orig.set_title('Original patch')
        ax_orig.axis('off')

        # — Run & show each augmentation in sequence —
        sample = {'image': patch_img, 'label': patch_lbl}
        for j, t in enumerate(tail_steps, start=2):
            out = t(sample)
            if isinstance(out, list): out = out[0]
            sample = out

            img_a = sample['image'].numpy()[0]  # H×W
            lbl_a = sample['label'].numpy()[0]
            ax = axes[i, j]
            ax.imshow(img_a, cmap='gray')
            ax.imshow(lbl_a, cmap='jet', alpha=0.3)
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
        five = train_pairs[:5]
        ph, pw = patch_cfg.get('size', (256, 256))
        visualize_augmentations_grid(five, train_t, patch_size=(ph, pw))
        sys.exit(0)

        print("[preprocess] Augmented transforms ready.")
