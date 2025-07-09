#!/usr/bin/env python3
"""
generate_snapshots.py

Generates static plot snapshots for each narration segment,
highlighting only the curves relevant to that segment.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# --- USER CONFIG ---------------------------------------------------------

ROOT_DIR = "study"           # your experiments folder
OUT_DIR  = "snapshots"       # where to save the PNGs
TAG      = "train/loss"      # which scalar to plot

# Define your narration segments and which curves to highlight.
# Use the exact labels that your experiments generate.
SEGMENTS = [
    {"name": "01_intro",   "highlight": []},
    {"name": "02_params",  "highlight": []},
    {"name": "03_segres",  "highlight": [
        "SegRes + Dice+CE + Full",
        "SegRes + Dice+CE + Patch"
    ]},
    {"name": "04_unet",    "highlight": [
        "UNet + Dice+CE + Full",
        "UNet + Dice+CE + Patch"
    ]},
    {"name": "05_loss_fn", "highlight": "all"},
    {"name": "06_summary", "highlight": "all"},
]

# Aesthetic settings
FIGSIZE = (10, 6)
FPS     = 10    # if you want to estimate how long each image shows


# --- HELPERS -------------------------------------------------------------

def find_experiments(root_dir):
    """Discover experiment folders and build human-readable labels."""
    ARCH_MAP  = {"unet":"UNet",   "segres":"SegRes"}
    LOSS_MAP  = {"dice_ce":"Dice+CE", "dice_focal":"Dice+Focal"}
    MODE_MAP  = {"full":"Full",  "patch":"Patch"}

    exps = []
    for d in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, d)
        if not os.path.isdir(path):
            continue
        logs = glob.glob(os.path.join(path, "outputs", "logs", "events.out.tfevents.*"))
        if not logs:
            continue
        try:
            arch, loss_key, mode = d.split("-")
        except ValueError:
            continue
        label = f"{ARCH_MAP.get(arch, arch)} + {LOSS_MAP.get(loss_key, loss_key)} + {MODE_MAP.get(mode, mode)}"
        exps.append({"name": d, "label": label, "files": logs})
    return exps

def load_curve(files, tag):
    """Load and merge scalar 'tag' from all event files."""
    entries = []
    for f in files:
        ea = event_accumulator.EventAccumulator(f,
            size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            entries.extend(ea.Scalars(tag))
    by_step = {e.step: e.value for e in entries}
    steps = sorted(by_step)
    vals  = [by_step[s] for s in steps]
    return np.array(steps), np.array(vals)

# --- MAIN ---------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    experiments = find_experiments(ROOT_DIR)
    if not experiments:
        print("❌ No experiments found under", ROOT_DIR)
        return

    # Load all curves
    curves = []
    max_step = 0
    for exp in experiments:
        steps, vals = load_curve(exp["files"], TAG)
        curves.append({"label": exp["label"], "steps": steps, "vals": vals})
        if steps.size and steps[-1] > max_step:
            max_step = steps[-1]

    # Pad shorter curves to max_step so all share the same x-axis
    for c in curves:
        if c["steps"].size == 0:
            c["steps"] = np.arange(max_step+1)
            c["vals"]  = np.zeros(max_step+1)
        elif c["steps"][-1] < max_step:
            pad = np.arange(c["steps"][-1]+1, max_step+1)
            c["steps"] = np.concatenate([c["steps"], pad])
            last_val = c["vals"][-1]
            c["vals"]  = np.concatenate([c["vals"], np.full_like(pad, last_val)])

    # For each segment, draw the plot and save a PNG
    for seg in SEGMENTS:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        highlight = seg["highlight"]

        for c in curves:
            lbl = c["label"]
            # decide if this curve is "on"
            if highlight == "all":
                is_on = True
            elif isinstance(highlight, list):
                is_on = (lbl in highlight)
            else:
                is_on = False

            if is_on:
                alpha, lw = 1.0, 2.5
            else:
                alpha, lw = 0.1, 1.0

            ax.plot(
                c["steps"],
                c["vals"],
                label=lbl,
                alpha=alpha,
                linewidth=lw
            )

        ax.set_xlim(0, max_step)
        ax.set_ylim(0, max(c["vals"].max() for c in curves) * 1.05)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(TAG.split("/")[-1].replace("_", " ").title())
        ax.set_title(f"{TAG.split('/')[-1].replace('_',' ').title()} — {seg['name']}")

        # single-column legend outside
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=1,
            fontsize="small",
            frameon=False
        )

        plt.tight_layout()
        outpath = os.path.join(OUT_DIR, f"{seg['name']}.png")
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"✔ Saved {outpath} (highlight={highlight})")

    print(f"\nGenerated {len(SEGMENTS)} snapshots in '{OUT_DIR}/' — ready for video stitching.")

if __name__ == "__main__":
    main()
