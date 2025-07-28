import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('./study-corrected/data.csv')

# Define consistent colors
colors = {
    'unet-dice_ce-full': 'tab:blue',
    'unet-dice_focal-full': 'tab:orange',
    'unet-dice_focal-patch': 'tab:green',
    'segres-dice_focal-patch': 'tab:red'
}

# Prepare legend labels with aligned columns using monospace padding
labels = {
    'unet-dice_ce-full':     f"{'Unet':<9}{'Dice CE':<12}{'Full image'}",
    'unet-dice_focal-full':  f"{'Unet':<9}{'Dice Focal':<12}{'Full image'}",
    'unet-dice_focal-patch': f"{'Unet':<9}{'Dice Focal':<12}{'Patch'}",
    'segres-dice_focal-patch': f"{'SegRes':<9}{'Dice Focal':<12}{'Patch'}"
}

# Common plot args
lw = 3
font_props = {'family': 'monospace', 'size': 12}
common_title = "Validation Loss for Retinal Vessel Segmentation"

# 1. Only Unet + Dice CE + Full
fig, ax = plt.subplots(figsize=(10,6), dpi=300)
ax.plot(df.index, df['unet-dice_ce-full_val_loss'], color=colors['unet-dice_ce-full'],
        linewidth=lw, label=labels['unet-dice_ce-full'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title(common_title)
title_string = "Network    Loss                Data  "
ax.legend(prop=font_props, title=title_string, title_fontsize=12, loc='best')

fig.tight_layout()
fig.savefig('./figures/1_val_loss_unet_diceCE_full.png')

# 2. CE de-emphasized, Focal (Full) highlighted
fig, ax = plt.subplots(figsize=(10,6), dpi=300)
ax.plot(df.index, df['unet-dice_ce-full_val_loss'], color=colors['unet-dice_ce-full'],
        linewidth=lw, alpha=0.3, label=labels['unet-dice_ce-full'])
ax.plot(df.index, df['unet-dice_focal-full_val_loss'], color=colors['unet-dice_focal-full'],
        linewidth=lw, label=labels['unet-dice_focal-full'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title(common_title)

ax.legend(prop=font_props, title=title_string, title_fontsize=12, loc='best')

fig.tight_layout()
fig.savefig('./figures/2_val_loss_unet_diceFocal_full.png')

# 3. Full curves de-emphasized, Focal (Patch) highlighted
fig, ax = plt.subplots(figsize=(10,6), dpi=300)
ax.plot(df.index, df['unet-dice_ce-full_val_loss'], color=colors['unet-dice_ce-full'],
        linewidth=lw, alpha=0.3, label=labels['unet-dice_ce-full'])
ax.plot(df.index, df['unet-dice_focal-full_val_loss'], color=colors['unet-dice_focal-full'],
        linewidth=lw, alpha=0.3, label=labels['unet-dice_focal-full'])
ax.plot(df.index, df['unet-dice_focal-patch_val_loss'], color=colors['unet-dice_focal-patch'],
        linewidth=lw, label=labels['unet-dice_focal-patch'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title(common_title)

ax.legend(prop=font_props, title=title_string, title_fontsize=12, loc='best')

fig.tight_layout()
fig.savefig('./figures/3_val_loss_unet_diceFocal_patch.png')

# 4. Highlight SegRes + Dice Focal (Patch)
fig, ax = plt.subplots(figsize=(10,6), dpi=300)
ax.plot(df.index, df['unet-dice_ce-full_val_loss'], color=colors['unet-dice_ce-full'],
        linewidth=lw, alpha=0.3, label=labels['unet-dice_ce-full'])
ax.plot(df.index, df['unet-dice_focal-full_val_loss'], color=colors['unet-dice_focal-full'],
        linewidth=lw, alpha=0.3, label=labels['unet-dice_focal-full'])
ax.plot(df.index, df['unet-dice_focal-patch_val_loss'], color=colors['unet-dice_focal-patch'],
        linewidth=lw, alpha=0.3, label=labels['unet-dice_focal-patch'])
ax.plot(df.index, df['segres-dice_focal-patch_val_loss'], color=colors['segres-dice_focal-patch'],
        linewidth=lw, label=labels['segres-dice_focal-patch'])
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title(common_title)

ax.legend(prop=font_props, title=title_string, title_fontsize=12, loc='best')

fig.tight_layout()
fig.savefig('./figures/4_val_loss_segres_diceFocal_patch.png')

plt.close('all')

