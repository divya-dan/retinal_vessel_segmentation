#!/usr/bin/env python3
"""
export_losses_wide.py

Scan a root directory of TensorBoard logs, pull out train/loss & val/loss
for every experiment and every epoch, then pivot to a wide CSV where each
experiment has its own train and val columns.
"""

import os
import glob
import argparse
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def find_experiments(root_dir):
    exps = {}
    for entry in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, entry)
        if not os.path.isdir(path):
            continue
        files = glob.glob(os.path.join(path, "outputs", "logs", "events.out.tfevents.*"))
        if files:
            exps[entry] = files
    return exps

def load_scalars(event_files, tag):
    """
    Load and merge scalar data for `tag` from a list of event files.
    Returns a dict: { step -> value }.
    """
    by_step = {}
    for fpath in event_files:
        ea = event_accumulator.EventAccumulator(
            fpath,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            continue
        for e in ea.Scalars(tag):
            by_step[e.step] = e.value
    return by_step

def main(root_dir, out_path):
    experiments = find_experiments(root_dir)
    if not experiments:
        print(f"No experiments found under {root_dir}")
        return

    # We'll build a list of per-experiment DataFrames, then outer-join them on 'epoch'
    df_list = []
    for exp_name, files in experiments.items():
        train = load_scalars(files, "train/loss")
        val   = load_scalars(files, "val/loss")

        # union of epochs
        all_epochs = sorted(set(train) | set(val))
        df_exp = pd.DataFrame({
            'epoch': all_epochs,
            f'{exp_name}_train_loss': [ train.get(e, pd.NA) for e in all_epochs ],
            f'{exp_name}_val_loss':   [   val.get(e, pd.NA) for e in all_epochs ],
        })
        df_list.append(df_exp)

    # merge all on 'epoch'
    from functools import reduce
    df_wide = reduce(lambda left, right: pd.merge(left, right, on='epoch', how='outer'),
                     df_list)

    df_wide = df_wide.sort_values('epoch').reset_index(drop=True)

    # handle output path
    if os.path.isdir(out_path) or out_path.endswith(os.sep):
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, "losses_wide.csv")

    df_wide.to_csv(out_path, index=False)
    print(f"✅ Exported wide-format losses to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Export train/val loss of all experiments to a wide CSV"
    )
    p.add_argument("root",
                   help="Root folder containing your experiment subdirectories")
    p.add_argument("-o", "--out",
                   default="losses_wide.csv",
                   help="Output CSV filename or directory")
    args = p.parse_args()

    main(args.root, args.out)
