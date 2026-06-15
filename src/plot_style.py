#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  plot_style.py -- shared "paper" plotting style for ToyNDFit, mirroring the
#  conventions used in GuNFlows make_paper_plots.py (serif/CM fonts, in-pointing
#  ticks, no frame on the legend, no titles). Use apply_paper_style() once, then
#  ax_fontsize() per-axes after layout. clean_name() strips '_' and '#' from
#  systematic-parameter names for axis labels / legends.
# =============================================================================
from __future__ import annotations
import re
import matplotlib.pyplot as plt


def apply_paper_style(usetex: bool = False) -> None:
    """Apply the paper rcParams globally. usetex=False keeps it machine-independent."""
    rc = {
        "text.usetex":          bool(usetex),
        "mathtext.fontset":     "cm",
        "font.family":          "serif",
        "font.size":            16,
        "axes.labelsize":       16,
        "xtick.labelsize":      14,
        "ytick.labelsize":      14,
        "legend.fontsize":      14,
        "legend.frameon":       False,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            False,
        "ytick.right":          False,
        "xtick.minor.visible":  False,
        "ytick.minor.visible":  False,
        "xtick.major.size":     7,
        "ytick.major.size":     7,
        "xtick.major.width":    1.4,
        "ytick.major.width":    1.4,
        "axes.linewidth":       1.4,
        "axes.labelpad":        10,
        "lines.linewidth":      2.0,
        "figure.dpi":           150,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
    }
    if usetex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    plt.rcParams.update(rc)


def ax_fontsize(ax, label_fs: int = 16, tick_fs=None, legend_fs=None) -> None:
    """Force explicit font sizes on an axes (paper convention)."""
    if tick_fs is None:
        tick_fs = label_fs - 2
    if legend_fs is None:
        legend_fs = label_fs - 2
    ax.xaxis.label.set_size(label_fs)
    ax.yaxis.label.set_size(label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(tick_fs)
    if ax.get_legend():
        for t in ax.get_legend().get_texts():
            t.set_fontsize(legend_fs)


def clean_name(name) -> str:
    """Human-readable systematic-parameter label: drop '_' and '#', collapse spaces.

    e.g. 'Cross-Section#MaCCQE' -> 'Cross-Section MaCCQE', 'norm_nu' -> 'norm nu'.
    """
    s = str(name).replace("#", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()
