#!/usr/bin/env python
import os
import sys

# Add project root to PYTHONPATH for imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
# Actually project root is one level up from src/
project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

from monai.data import CacheDataset, SmartCacheDataset, DataLoader
from src.config import load_config
from src.data.preprocess import split_train_val, get_transforms, get_image_mask_pairs

"""
Dataset wrapper for retinal vessel segmentation.
- Loads train/val/test pairs, applies transforms, and returns DataLoaders.
"""

def get_dataloaders():
    # Load config parameters
    cfg = load_config()
    data_root = cfg['data']['data_root']
    train_folder = os.path.join(data_root, 'train')
    test_folder = os.path.join(data_root, 'test')
    split_ratio = cfg['data']['train_val_split']
    seed = cfg['data']['random_seed']
    batch_size = cfg['train']['batch_size']
    num_workers = cfg.get('data', {}).get('n_workers', os.cpu_count())
    image_size = cfg['data'].get('image_size', (512, 512))
    use_patches = cfg.get('patch', {}).get('use', False)

    # Get train/val splits
    train_pairs, val_pairs = split_train_val(train_folder, split_ratio, seed)
    # Get test pairs (no split)
    test_pairs = get_image_mask_pairs(test_folder)

    # Create transforms
    train_transforms, val_transforms, test_transforms = get_transforms(image_size=image_size, use_patches=use_patches)

    # Build datasets
    DatasetClass = SmartCacheDataset if use_patches else CacheDataset

    # SmartCacheDataset doesn't accept num_workers parameter
    if use_patches:
        train_ds = DatasetClass(data=train_pairs, transform=train_transforms, cache_rate=1.0)
    else:
        train_ds = DatasetClass(data=train_pairs, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
    val_ds   = CacheDataset(data=val_pairs,   transform=val_transforms,   cache_rate=1.0, num_workers=num_workers)
    test_ds  = CacheDataset(data=test_pairs,  transform=test_transforms,  cache_rate=1.0, num_workers=num_workers)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[dataset] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples, Test: {len(test_ds)} samples")
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Example usage
    train_loader, val_loader, test_loader = get_dataloaders()
    for batch in train_loader:
        images = batch['image']  # shape: [B, C, H, W]
        labels = batch['label']  # shape: [B, C, H, W]
        print(f"Batch images: {images.shape}, labels: {labels.shape}")
        break
