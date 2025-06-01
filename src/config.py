import os
import yaml

# Updated default configuration
default_config = {
    'data': {
        'data_root': 'data',              # Root folder for all datasets
        'train_val_split': 0.1,           # Fraction of training set to use for validation
        'random_seed': 42,                # For reproducibility
        'image_size': '(512, 512)',       # Resize images to this size
    },
    'patch': {
        'use': False,                    # Toggle to use patch-based training
        'size': '(256, 256)',              # Patch size for training
        'pos': 1,                        # Number of positive samples
        'neg': 1,                        # Number of negative samples
        'num_samples': 4,                # Total patches per image
    },
    'sliding_window': {
        'roi_size': '(256, 256)',          # Size of sliding window patches
        'sw_batch_size': 8,              # Batch size for sliding window inference
        'overlap': 0.25,                 # Overlap ratio for sliding window
    },
    'model': {
        'unet': {
            'in_channels': 3,             # RGB fundus images
            'out_channels': 1,            # Binary vessel mask
            'channels': [16, 32, 64, 128, 256],  # Feature maps
            'strides': [2, 2, 2, 2],       # Downsampling steps
            'num_res_units': 2,           # Number of residual blocks
            'norm': 'BATCH',              # Normalization type
        }
    },
    'train': {
        'multi_gpu': False,
        'device': 'cuda',                 # Options: 'cuda' or 'cpu'
        'seed': 42,
        'learning_rate': 1e-4,
        'batch_size': 8,
        'num_epochs': 50,
        'val_interval': 1,                # Validate every N epochs
        'loss_function': 'dice_ce',         # Options: 'dice_ce' or 'dice_focal'
    },
    'optimizer': {
        'type': 'Adam',                   # Optimizer type
        'weight_decay': 1e-5,
    },
    'scheduler': {
        'step_size': 10,                  # Reduce LR every N epochs
        'gamma': 0.5,                     # LR decay rate
    },
    'early_stopping': {
        'metric': 'val_dice',             # Metric to monitor
        'patience': 10,                   # Early stop after N stagnant epochs
    },
    'paths': {
        'checkpoint_dir': 'models',       # Where to save checkpoints
        'log_dir': 'outputs/logs',        # For TensorBoard
        'output_dir': 'outputs',          # Visuals, metrics, etc.
    }
}


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """
    Load the YAML configuration file. If it doesn't exist, create one with defaults.
    Returns a dict of configuration parameters.
    """
    # Use absolute paths to avoid confusion
    if not os.path.isabs(config_path):
        # Find project root (parent directory of src/)
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False, width=80)
        print(f"[config] Created default config at {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"[config] Loaded configuration from {config_path}")
    return cfg


if __name__ == '__main__':
    config = load_config()
    import pprint; pprint.pprint(config)
