#!/usr/bin/env python3
"""
compare_tensorboard.py

Scan a root directory of TensorBoard logs, extract specified scalars
from each experiment, and plot comparative curves with differentiated styles.
"""

import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# metrics to plot
TAGS = [
    "train/loss",
    "train/learning_rate",
    "val/loss",
    "val/dice",
]

# map architecture name -> marker
ARCH_MARKERS = {
    "unet": "o",
    "segres": "s",
}

def find_event_files(root_dir):
    experiments = {}
    for exp_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        pattern = os.path.join(exp_dir, "outputs", "logs", "events.out.tfevents.*")
        files = glob.glob(pattern)
        if files:
            experiments[os.path.basename(exp_dir)] = files
    return experiments

def load_scalars(event_files, tag):
    all_entries = []
    for f in event_files:
        ea = event_accumulator.EventAccumulator(f,
            size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            continue
        all_entries.extend(ea.Scalars(tag))
    # merge by step
    by_step = {e.step: e.value for e in all_entries}
    steps = sorted(by_step.keys())
    values = [by_step[s] for s in steps]
    return steps, values

def plot_comparisons(root_dir, out_dir="comparisons", marker_interval=5):
    os.makedirs(out_dir, exist_ok=True)
    experiments = find_event_files(root_dir)
    if not experiments:
        print(f"No event files found under {root_dir}")
        return

    for tag in TAGS:
        plt.figure(figsize=(8,6))
        for exp_name, files in experiments.items():
            steps, vals = load_scalars(files, tag)
            if not steps:
                continue

            # linestyle: solid for full, dashed for patch
            linestyle = "-" if exp_name.endswith("full") else "--"
            # architecture = first token: 'unet' or 'segres'
            arch = exp_name.split("-")[0]
            marker = ARCH_MARKERS.get(arch, None)

            plt.plot(
                steps,
                vals,
                label=exp_name,
                linestyle=linestyle,
                marker=marker,
                markevery=marker_interval,
                markersize=4,
                linewidth=1.5
            )

        # titles & labels
        pretty_tag = tag.split("/")[-1].replace("_", " ").title()
        plt.title(pretty_tag)
        plt.xlabel("Epoch")
        plt.ylabel(pretty_tag)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize="small", loc="best")

        # save
        fname = f"{tag.replace('/', '_')}.png"
        outpath = os.path.join(out_dir, fname)
        plt.savefig(outpath, bbox_inches="tight")
        print(f"→ {outpath}")
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare TensorBoard scalars across experiments with styled lines"
    )
    parser.add_argument("root", help="Root directory of your experiments")
    parser.add_argument(
        "--out", "-o",
        default="comparisons",
        help="Where to save the plots"
    )
    parser.add_argument(
        "--markerevery", "-m",
        type=int,
        default=5,
        help="Place a marker every N epochs"
    )
    args = parser.parse_args()
    plot_comparisons(args.root, args.out, marker_interval=args.markerevery)
