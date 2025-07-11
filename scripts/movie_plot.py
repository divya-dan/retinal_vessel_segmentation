import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('./study/losses_wide.csv')

# Prepare mapping for labels
label_map = {
    'segres-dice_ce-full_val_loss': 'SegResNet + Dice+CE (Full)',
    'segres-dice_ce-patch_val_loss': 'SegResNet + Dice+CE (Patch)',
    'segres-dice_focal-full_val_loss': 'SegResNet + Dice+Focal (Full)',
    'segres-dice_focal-patch_val_loss': 'SegResNet + Dice+Focal (Patch)',
    'unet-dice_ce-full_val_loss': 'U-Net + Dice+CE (Full)',
    'unet-dice_ce-patch_val_loss': 'U-Net + Dice+CE (Patch)',
    'unet-dice_focal-full_val_loss': 'U-Net + Dice+Focal (Full)',
    'unet-dice_focal-patch_val_loss': 'U-Net + Dice+Focal (Patch)',
}

# Plot validation loss curves
plt.figure(figsize=(12, 8))

for col, label in label_map.items():
    # Determine marker based on model type
    
    if 'unet' in col.lower():
        marker = 's'
    elif 'segres' in col.lower():
        marker = 'o'
    else:
        marker = 'x'  # square as default
    
    # Determine line style based on data type
    if 'full' in col.lower():
        linestyle = '-'  # solid line
    elif 'patch' in col.lower():
        linestyle = '--'  # dashed line
    else:
        linestyle = '-'  # solid as default
    
    plt.plot(df['epoch'], df[col], label=label, marker=marker, linestyle=linestyle, markersize=5,  markevery=5)
    
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss over Training Epochs for Different Models, Data Types, and Loss Functions')
plt.legend(ncol=2, handlelength=5)  # Increase line length in legend
plt.grid(True, alpha=0.75)
plt.tight_layout()
# plt.show()
plt.savefig('./figures/movie-plot-1.png', dpi=300, bbox_inches='tight')
plt.close()