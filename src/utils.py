import uproot
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
from neut_codes import neut_reactions, neut_compact, neut_group
from nuisance_event import NUISANCEEvent, NUISANCEFile, SampleCodes

def to_1d_numpy(lst, dtype, float_fill=np.nan, int_fill=-999):
    # detect nested elements
    if any(isinstance(x, (list, tuple, np.ndarray)) for x in lst):
        raise ValueError("Branch contains nested/variable-length elements; flatten or convert to fixed-size first.")
    if np.issubdtype(dtype, np.floating):
        arr = np.array([float_fill if x is None else x for x in lst], dtype=dtype)
    else:
        # integer branch: replace None with sentinel
        arr = np.array([int_fill if x is None else x for x in lst], dtype=dtype)
    return arr

def find_bin(xedges, yedges, x, y):
    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1

    # Handle underflow/overflow:
    ix = np.clip(ix, 0, len(xedges) - 2)
    iy = np.clip(iy, 0, len(yedges) - 2)
    return ix, iy


def smear(Pmu, CosThetamu, pmu_map, cthetamap):
    """
    Smear the muon momentum and cosTheta with a Gaussian resolution from 2D histograms.
    The 2D histograms are obtained from ND280 MC truth-reco comparisons.

    Parameters
    ----------
    Pmu : float
        True muon momentum (GeV/c).
    CosThetamu : float
        True muon cosine of the angle.

    Returns
    -------
    Pmu_smeared : float
        Smeared muon momentum (GeV/c).
    CosThetamu_smeared : float
        Smeared muon cosine of the angle.
    """

    if Pmu is None or CosThetamu is None:
        return None, None

    # get resolution from maps
    xedges = pmu_map.axis(0).edges()
    yedges = pmu_map.axis(1).edges()
    # check that pmu and costhetamu maps have the same binning
    if not np.array_equal(xedges, cthetamap.axis(0).edges()) or not np.array_equal(yedges, cthetamap.axis(1).edges()):
        raise ValueError("pmu_map and cthetamap have different binning.")

    pmu_values = pmu_map.values()  # 2D numpy array with shape (nx, ny)
    cthetamu_values = cthetamap.values()  # 2D numpy array with shape (nx, ny)

    pmu_bin, costhetamu_bin = find_bin(pmu_map.axes[0].edges(), pmu_map.axes[1].edges(), Pmu, CosThetamu)

    pmu_resolution = pmu_values[pmu_bin, costhetamu_bin]
    ctheta_resolution = cthetamu_values[pmu_bin, costhetamu_bin]

    # smear with Gaussian
    Pmu_smeared = np.random.normal(Pmu, pmu_resolution * Pmu)
    CosThetamu_smeared = np.random.normal(CosThetamu, ctheta_resolution)

    # if (CosThetamu> 0.9):
    #     print(f"DEBUG: CosThetamu={CosThetamu}, ctheta_res={ctheta_resolution} => CosThetamu_smeared={CosThetamu_smeared}")

    # ensure CosThetamu_smeared is within [-1, 1]
    CosThetamu_smeared = max(-1.0, min(1.0, CosThetamu_smeared))

    return Pmu_smeared, CosThetamu_smeared


