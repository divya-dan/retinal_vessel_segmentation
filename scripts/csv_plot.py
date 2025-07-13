import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('./study/losses_wide.csv')

# Ensure output directories exist
os.makedirs('./figures', exist_ok=True)

# Mapping for validation loss labels
val_label_map = {
    'segres-dice_ce-full_val_loss':    'SegResNet + Dice+CE (Full – val)',
    'segres-dice_ce-patch_val_loss':   'SegResNet + Dice+CE (Patch – val)',
    'segres-dice_focal-full_val_loss':  'SegResNet + Dice+Focal (Full – val)',
    'segres-dice_focal-patch_val_loss': 'SegResNet + Dice+Focal (Patch – val)',
    'unet-dice_ce-full_val_loss':      'U-Net + Dice+CE (Full – val)',
    'unet-dice_ce-patch_val_loss':     'U-Net + Dice+CE (Patch – val)',
    'unet-dice_focal-full_val_loss':   'U-Net + Dice+Focal (Full – val)',
    'unet-dice_focal-patch_val_loss':  'U-Net + Dice+Focal (Patch – val)',
}

# Mapping for training loss labels
train_label_map = {
    'segres-dice_ce-full_train_loss':    'SegResNet + Dice+CE (Full – train)',
    'segres-dice_ce-patch_train_loss':   'SegResNet + Dice+CE (Patch – train)',
    'segres-dice_focal-full_train_loss':  'SegResNet + Dice+Focal (Full – train)',
    'segres-dice_focal-patch_train_loss': 'SegResNet + Dice+Focal (Patch – train)',
    'unet-dice_ce-full_train_loss':      'U-Net + Dice+CE (Full – train)',
    'unet-dice_ce-patch_train_loss':     'U-Net + Dice+CE (Patch – train)',
    'unet-dice_focal-full_train_loss':   'U-Net + Dice+Focal (Full – train)',
    'unet-dice_focal-patch_train_loss':  'U-Net + Dice+Focal (Patch – train)',
}

def plot_loss(label_map, suffix, title, outfile):
    plt.figure(figsize=(12, 8))
    for col, label in label_map.items():
        if not col.endswith(suffix):
            continue
        # markers by model
        marker = 's' if 'unet' in col else 'o'
        # linestyle by data type
        linestyle = '-' if 'full' in col else '--'
        plt.plot(df['epoch'], df[col],
                 label=label,
                 marker=marker,
                 linestyle=linestyle,
                 markersize=5,
                 markevery=10)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend(ncol=2, handlelength=5)
    plt.grid(True, alpha=0.75)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{title} plot saved to {outfile}")

# Plot validation losses (existing)
plot_loss(val_label_map, 'val_loss',
          'Validation Loss over Epochs',
          './figures/validation_loss.png')

# Plot training losses (new)
plot_loss(train_label_map, 'train_loss',
          'Training Loss over Epochs',
          './figures/training_loss.png')

# Build summary of best (min) losses for validation (existing)
val_records = []
for col, label in val_label_map.items():
    if col in df.columns:
        idx = df[col].idxmin()
        model_name = 'SegResNet' if 'segres' in col else 'U-Net'
        data_name = 'Full images' if 'full' in col else 'Patch'
        loss_name = 'Dice+CE' if 'dice_ce' in col else 'Dice+Focal'
        val_records.append({
            'Model': model_name,
            'Loss Fn': loss_name,
            'Data': data_name,
            'Min Val Loss': df.at[idx, col],
            'Epoch at Min': int(df.at[idx, 'epoch'])
        })
summary_val_df = pd.DataFrame(val_records)
print("\n### Validation Summary of Minimum Losses\n")
print(summary_val_df.to_markdown(index=False))

# Build summary of best (min) losses for training (new)
train_records = []
for col, label in train_label_map.items():
    if col in df.columns:
        idx = df[col].idxmin()
        model_name = 'SegResNet' if 'segres' in col else 'U-Net'
        data_name = 'Full images' if 'full' in col else 'Patch'
        loss_name = 'Dice+CE' if 'dice_ce' in col else 'Dice+Focal'
        train_records.append({
            'Model': model_name,
            'Loss Fn': loss_name,
            'Data': data_name,
            'Min Train Loss': df.at[idx, col],
            'Epoch at Min': int(df.at[idx, 'epoch'])
        })
summary_train_df = pd.DataFrame(train_records)
print("\n### Training Summary of Minimum Losses\n")
print(summary_train_df.to_markdown(index=False))
