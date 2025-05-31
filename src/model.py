#!/usr/bin/env python
import os
import sys

import torch
import torch.nn as nn
from monai.networks.nets import UNet

# Add project root to PYTHONPATH for imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
# Actually project root is one level up from src/
project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

from src.config import load_config


def get_unet_model(config):
    """
    Instantiate a MONAI 2D U-Net model based on config.yaml settings.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        torch.nn.Module: Initialized U-Net model
    """
    net_cfg = config['model']['unet']

    model = UNet(
        spatial_dims=2,
        in_channels=net_cfg['in_channels'],
        out_channels=net_cfg['out_channels'],
        channels=net_cfg['channels'],        # e.g., [16, 32, 64, 128, 256]
        strides=net_cfg['strides'],          # e.g., [2, 2, 2, 2]
        num_res_units=net_cfg['num_res_units'],
        norm=net_cfg.get('norm', 'BATCH')    # Options: BATCH, INSTANCE, etc.
    )

    return model


if __name__ == '__main__':
    cfg = load_config()
    model = get_unet_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optional: wrap in DataParallel for multi-GPU
    if cfg['train'].get('multi_gpu', False) and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Print model summary
    print(model)
    x = torch.randn(1, cfg['model']['unet']['in_channels'], 512, 512).to(device)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
