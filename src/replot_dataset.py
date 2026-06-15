#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  replot_dataset.py
#  Re-make the dataset plots from the ROOT output of produce_dataset_from_NUISANCE.py
#  WITHOUT re-running the (expensive) dataset generation. Reads the "sample_sum"
#  TTree (scalar branches SelectedSample/Pmu/CosThetamu/Reaction + the per-event
#  spline TGraph branches) and reproduces:
#     - the kinematic distributions (plot_utils.plot_samples)
#     - the per-event spline weight plots w(eta)   (plot_utils.plot_spline_weight)
#  using the same uniform paper style as the producer.
#
#  Usage:
#    python replot_dataset.py -i dataset_mc.root -o ./plots [-n 10]
# =============================================================================
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import ROOT

import plot_utils

SCALAR_BRANCHES = ["SelectedSample", "Pmu", "CosThetamu", "Reaction"]


def main():
    p = argparse.ArgumentParser(description="Re-plot dataset plots from a produce_dataset_from_NUISANCE.py ROOT file.")
    p.add_argument("-i", "--input_root", required=True, help="MC dataset ROOT file (tree 'sample_sum').")
    p.add_argument("-o", "--out_dir", default="./plots_replot", help="Output directory for the plots.")
    p.add_argument("-t", "--tree", default="sample_sum", help="TTree name (default: sample_sum).")
    p.add_argument("-n", "--n_spline_events", type=int, default=10,
                   help="Number of leading events for the per-event spline weight plots.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    f = ROOT.TFile.Open(args.input_root)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {args.input_root}")
    tree = f.Get(args.tree)
    if not tree:
        raise RuntimeError(f"Tree '{args.tree}' not found in {args.input_root}")

    # spline branches are stored as TClonesArray of TGraph
    spline_branches = [b.GetName() for b in tree.GetListOfBranches()
                       if b.GetClassName() == "TClonesArray"]
    present_scalars = {b.GetName() for b in tree.GetListOfBranches()}
    print(f"Entries: {tree.GetEntries()}  spline branches: {spline_branches}", flush=True)

    SelectedSample, Pmu, CosThetamu, Reaction = [], [], [], []
    spline_xy = {name: [] for name in spline_branches}  # name -> list of (event_idx, x, y)

    n_spl = int(args.n_spline_events)
    for idx, ev in enumerate(tree):
        SelectedSample.append(int(getattr(ev, "SelectedSample")))
        Pmu.append(float(getattr(ev, "Pmu")))
        CosThetamu.append(float(getattr(ev, "CosThetamu")))
        Reaction.append(int(getattr(ev, "Reaction")))
        if idx < n_spl:
            for name in spline_branches:
                tca = getattr(ev, name)
                if tca is None or tca.GetEntries() == 0:
                    continue
                g = tca.At(0)
                if g is None or g.GetN() == 0:
                    continue
                x = np.array([g.GetX()[j] for j in range(g.GetN())], dtype=float)
                y = np.array([g.GetY()[j] for j in range(g.GetN())], dtype=float)
                spline_xy[name].append((idx, x, y))

    # 1) kinematic distributions
    plot_utils.plot_samples(SelectedSample, Pmu, CosThetamu, Reaction, out_dir=args.out_dir)

    # 2) per-event spline weight plots w(eta)
    n_written = 0
    for name, items in spline_xy.items():
        for (idx, x, y) in items:
            plot_utils.plot_spline_weight(
                x, y, name, os.path.join(args.out_dir, f"spline_{name}_event_{idx}.png"))
            n_written += 1
    print(f"Wrote kinematic plots + {n_written} spline weight plots to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
