import uproot
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
from scipy import interpolate
from utils import *
import plot_style


def plot_spline_weight(x, y, name, out_path):
    """Paper-style plot of a per-event spline weight w(eta) vs the dial eta.

    x : dial values (eta), y : weights w. `name` is the systematic-parameter name
    (cleaned of '_' and '#' for the legend). No title; ylabel = w(eta).
    """
    plot_style.apply_paper_style()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    try:
        tck, _u = interpolate.splprep([x, y], s=0, k=3)
        unew = np.arange(0, 1.01, 0.01)
        out = interpolate.splev(unew, tck)
        ax.plot(out[0], out[1])
    except Exception:
        pass
    ax.plot(x, y, marker="o", linestyle="None", label=plot_style.clean_name(name))
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$w(\eta)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_style.ax_fontsize(ax)
    fig.savefig(out_path)
    plt.close(fig)

def plot_samples(SelectedSample_list, Pmu_list, CosThetamu_list, Reaction_list, out_dir="./"):
    """
    Plot the kinematic distributions for each sample in SelectedSample_list.
    Uniform paper style (plot_style.apply_paper_style): no titles, serif fonts.
    """
    print("Plotting kinematic distributions...")
    plot_style.apply_paper_style()
    os.makedirs(out_dir, exist_ok=True)

    unique, counts = np.unique(SelectedSample_list, return_counts=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.bar(unique, counts)
    ax.set_xlabel("Selected sample")
    ax.set_ylabel("Events")
    ax.set_xticks(unique)
    plot_style.ax_fontsize(ax)
    fig.savefig(f"{out_dir}/SelectedSample_distribution.png")
    plt.close(fig)

    for sample_code in unique:
        indices = [i for i, x in enumerate(SelectedSample_list) if x == sample_code]
        # Group by reaction for this sample
        reaction_vals = [Reaction_list[i] for i in indices]
        Pmu_vals = [Pmu_list[i] for i in indices]
        CosThetamu_vals = [CosThetamu_list[i] for i in indices]

        unique_reactions, _ = np.unique(reaction_vals, return_counts=True)
        stacked_data = []
        labels = []
        for r in unique_reactions:
            pmus_for_r = [p for p, rv in zip(Pmu_vals, reaction_vals) if rv == r and p is not None]
            if len(pmus_for_r) > 0:
                stacked_data.append(pmus_for_r)
                labels.append(f"{neut_compact[r]}")

        if len(stacked_data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.hist(stacked_data, bins=50, stacked=True, alpha=0.8, label=labels)
        ax.set_xlabel(r"$p_\mu$ [GeV/$c$]")
        ax.set_ylabel("Events")
        ax.legend(fontsize="small")
        plot_style.ax_fontsize(ax)
        fig.savefig(f"{out_dir}/Pmu_distribution_sample_{sample_code}.png")
        plt.close(fig)

        stacked_data_ctheta = []
        labels_ctheta = []
        for r in unique_reactions:
            cthetas_for_r = [ct for ct, rv in zip(CosThetamu_vals, reaction_vals) if rv == r and ct is not None]
            if len(cthetas_for_r) > 0:
                stacked_data_ctheta.append(cthetas_for_r)
                labels_ctheta.append(f"{neut_compact[r]}")

        if len(stacked_data_ctheta) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.hist(stacked_data_ctheta, bins=50, stacked=True, alpha=0.8, label=labels_ctheta, range=(-1, 1))
        ax.set_xlabel(r"$\cos\theta_\mu$")
        ax.set_ylabel("Events")
        ax.legend(fontsize="small")
        plot_style.ax_fontsize(ax)
        fig.savefig(f"{out_dir}/CosThetamu_distribution_sample_{sample_code}.png")
        plt.close(fig)


def plot_nu_energy(file_path, tree_name="FlatTree_VARS"):
    """
    Plot the neutrino energy distribution from the NUISANCE flat tree.
    """
    print("Plotting neutrino energy distribution...")

    f = NUISANCEFile(file_path, tree_name, relevant_keys=None, step=10000)
    num_entries = len(f)
    Px_nu_list = []
    Py_nu_list = []
    Pz_nu_list = []

    pbar = tqdm(total=1000)
    for i, event in enumerate(f):
        if (i + 1) % (num_entries // 1000) == 0:
            pbar.update(1)
        # if (i + 1) % 20000 == 0:
        #     print(f"Read {i + 1}/{num_entries} entries...")
        nu_mask = (np.array(event.pdg_init) == 14)
        if np.any(nu_mask):
            Px_nu = np.array(event.px_init)[nu_mask]
            Py_nu = np.array(event.py_init)[nu_mask]
            Pz_nu = np.array(event.pz_init)[nu_mask]
            if (len(Px_nu) > 1):
                print(f"Found {len(Px_nu)} neutrinos in event {i + 1}")
            elif(len(Px_nu) == 1):
                Px_nu_list.append(Px_nu[0])
                Py_nu_list.append(Py_nu[0])
                Pz_nu_list.append(Pz_nu[0])


    pbar.close()

    # draw histograms for Px_nu, Py_nu, Pz_nu
    plt.hist(Px_nu_list, bins=50, alpha=0.7)
    plt.xlabel('Px_nu (GeV/c)')
    plt.ylabel('Number of Events')
    plt.title('Neutrino Px Distribution')
    plt.savefig('test/Px_nu_distribution.png')
    plt.clf()
    plt.hist(Py_nu_list, bins=50, alpha=0.7)
    plt.xlabel('Py_nu (GeV/c)')
    plt.ylabel('Number of Events')
    plt.title('Neutrino Py Distribution')
    plt.savefig('test/Py_nu_distribution.png')
    plt.clf()
    plt.hist(Pz_nu_list, bins=50, alpha=0.7)
    plt.xlabel('Pz_nu (GeV/c)')
    plt.ylabel('Number of Events')
    plt.title('Neutrino Pz Distribution')
    plt.savefig('test/Pz_nu_distribution.png')
    plt.clf()

