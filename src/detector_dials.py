"""Detector-response dials (type: detector), ND280-style variations of the reconstructed muon kinematics, exact for this toy's
detector model: reco p ~ N(p, r_p(p, cos) p), reco cos ~ N(cos, r_c(p, cos)) clipped to [-1, 1] (utils.smear, ND280 resolution
maps, bins of the TRUE p and cos). A variation changes that Gaussian; the event weight is the ratio of the density of the event's
stored reco value under the varied and the nominal Gaussian. ND280's published forms (arXiv:1502.01550 Sec. IV.B.4; C. Wret,
T2K-THESIS-097 Eqs. 5.9-5.15) map onto it as:
  mom_scale   reco p' = (1 + e) reco p             -> N((1 + e) p, (1 + e) s_p)   (magnetic field strength)
  mom_shift   reco p' = reco p + d                 -> N(p + d, s_p)               (momentum bias from vertex migration)
  mom_resol   reco p' = p + k (reco p - p)         -> N(p, k s_p)                 (in a range of true p and cos)
  angle_resol reco c' = c + k (reco c - c)         -> N(c, k s_c)
ND280 smears 1/p_t; this toy smears p, so the resolution dials act on p. Events whose reco cos is clipped to +-1 fall outside every
analysis bin and get no response."""
import numpy as np
from utils import find_bin


def resolutions(p, c, pmu_map, cthetamap):
    """r_p and r_c of an event, looked up at its true p and cos exactly as utils.smear does (ND280 resolution maps, reso_maps/)."""
    ix, iy = find_bin(pmu_map.axes[0].edges(), pmu_map.axes[1].edges(), p, c)
    return pmu_map.values()[ix, iy], cthetamap.values()[ix, iy]


def _ratio(x, m1, s1, m0, s0):
    """Ratio of the Gaussian densities N(x; m1, s1) / N(x; m0, s0): the exact event weight of a changed smearing."""
    return (s0 / s1) * np.exp(-0.5 * ((x - m1) / s1) ** 2 + 0.5 * ((x - m0) / s0) ** 2)


def weights(item, p, c, p_reco, c_reco, r_p, r_c):
    """Weights at the dial's knots for one event (p in MeV/c), or None where the dial does not act."""
    if p is None or p_reco is None:
        return None
    k, kind, s_p = np.asarray(item["knots"], float), item["response"], r_p * p
    # DET_mom_scale: momentum scale from the magnetic field strength, 0.6 %, T2K arXiv:1502.01550 Sec. IV.B.4 and arXiv:1403.2552 Sec. VI.A.1; form C. Wret, T2K-THESIS-097 (2016) Eq. 5.15
    if kind == "mom_scale":
        return _ratio(p_reco, (1 + k) * p, (1 + k) * s_p, p, s_p)
    # DET_mom_bias_vertex: momentum bias from vertex migration, 7 MeV/c, T2K arXiv:1801.05148 Sec. III.D, Table III
    if kind == "mom_shift":
        return _ratio(p_reco, p + k, s_p, p, s_p)
    # DET_mom_resol_*: TPC momentum resolution scale per true-momentum bin (form: Wret, T2K-THESIS-097 Eq. 5.9; data/MC T2K arXiv:1403.2552; per-bin 10 %: A. Garcia Soto, PhD thesis UAB (2017) Sec. 9.3.1); high-angle tracks: T2K arXiv:1801.05148
    if kind == "mom_resol":
        (plo, phi), (clo, chi) = item["p_range"], item.get("cos_range", [-1.0, 1.0])
        return _ratio(p_reco, p, k * s_p, p, s_p) if plo <= p < phi and clo <= c < chi else None
    # DET_angle_resol: muon angle resolution scale, 10 % (no public ND280 number)
    if kind == "angle_resol":
        return _ratio(c_reco, c, k * r_c, c, r_c) if abs(c_reco) < 1 else None
    raise ValueError(f"unknown detector response {kind}")
