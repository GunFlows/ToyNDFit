"""Stress-test dials (type: stress): constructed responses, no publication, that give the likelihood features a fit or a normalizing
flow finds hard: a parameter on its bound (PB_like), a non-Gaussian enhancement (A2p2h_gauss), a kink at the nominal value
(CC1pi_kink), a moving threshold (RES_lowQ2_thr), a response even in the parameter, with two islands (DIS_even). The prototype's
full set (GuNFlows variants/proto and nl76) also has a dipole axial mass on CCQE (MA_QE) and two normalisations, Norm_CCcoh and
Norm_DIS (`normalisations` of the configuration: applied by GUNDAM through their apply_condition, no per-event graph).

Each dial acts on the events of its NEUT modes (`modes`, negative for antineutrinos) through their true kinematics in the NUISANCE
flat tree (Q2 in GeV^2, q0 and q3 in GeV, Enu_true in GeV). The producer turns the weights at the knots into the per-event graphs,
and weights the pseudo-data with every dial at its `truth` (the data tree's `weight` branch). Parameters per dial in
config/config_stress_dials.yaml.

Created for the GuNFlows scaling study, October 2026 (the prototype's dials, formerly GuNFlows studies/proto/proto_dials.py).
"""
import numpy as np
import uproot

KEYS = ["Mode", "Q2", "q0", "q3", "Enu_true"]


def load(file_path, entry_start, entry_stop, tree_name="FlatTree_VARS"):
    """Mode and true kinematics (float64) of the entries [entry_start, entry_stop) of the flat tree, in file order (the producer's order)."""
    a = uproot.open(file_path)[tree_name].arrays(KEYS, entry_start=entry_start, entry_stop=entry_stop, library="np")
    return {k: a[k].astype(np.float64) for k in KEYS}


def t_lin(k, kmin, kmax):
    """k mapped linearly onto [-1, 1] between kmin and kmax, clipped outside."""
    return np.clip(2 * (k - kmin) / (kmax - kmin) - 1, -1, 1)


def logistic(x):
    """1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def pb_like(eta, ev):
    """Pauli-blocking-like suppression at low q3: w = 1 - eta (1 - q3/0.3)^2 below q3 = 0.3 GeV, eta in [0, 1]."""
    q3 = ev["q3"]
    shape = np.where(q3 < 0.3, (1 - q3 / 0.3) ** 2, 0.0)
    return np.clip(1 - eta * shape, 0, None)


def a2p2h_gauss(A, ev):
    """MINERvA-like enhancement in (q0, q3): w = 1 + A G, G a Gaussian of widths 0.06 and 0.13 GeV centred at (0.25, 0.50) GeV."""
    g = np.exp(-0.5 * ((ev["q0"] - 0.25) / 0.06) ** 2 - 0.5 * ((ev["q3"] - 0.50) / 0.13) ** 2)
    return 1 + A * g


def ma_qe_dipole(ma, ev):
    """Dipole axial mass: w = [(1 + Q2/1.03^2) / (1 + Q2/ma^2)]^4, relative to the nominal 1.03 GeV (Q2 clipped at 0, ma at 0.05 GeV)."""
    q2 = np.clip(ev["Q2"], 0, None)
    return ((1 + q2 / 1.03 ** 2) / (1 + q2 / np.maximum(ma, 0.05) ** 2)) ** 4


def norm(x, ev):
    """Normalisation: w = x (applied by GUNDAM through the dial's apply_condition; used here for the pseudo-data's weights)."""
    return x * np.ones_like(ev["Q2"])


def cc1pi_kink(eta, ev):
    """Kink at the nominal eta = 1: w = 1 + t (eta - 1) times 0.6 above and 0.2 below, t = Enu mapped onto [-1, 1] over 0.3-3 GeV."""
    d = eta - 1.0
    t = t_lin(ev["Enu_true"], 0.3, 3.0)
    return np.clip(1 + t * np.where(d > 0, 0.6 * d, 0.2 * d), 0, None)


def res_lowq2_thr(eta, ev):
    """Moving threshold: a 50 % suppression below Q2_c = 0.1 eta GeV^2 (logistic edge of width 0.01 GeV^2), relative to the nominal
    threshold (eta = 1)."""
    supp = lambda e: 1 - 0.5 * logistic((0.1 * e - ev["Q2"]) / 0.01)
    return supp(eta) / supp(1.0)


def dis_even(eta, ev):
    """Even response, only eta^2 enters (+eta and -eta give the same weight): w = 1 + 0.15 t eta^2, t = Enu mapped onto [-1, 1]
    over 0.5-5 GeV."""
    return np.clip(1 + 0.15 * t_lin(ev["Enu_true"], 0.5, 5.0) * eta ** 2, 0, None)


RESPONSES = {"pb_like": pb_like, "a2p2h_gauss": a2p2h_gauss, "ma_qe_dipole": ma_qe_dipole, "cc1pi_kink": cc1pi_kink,
             "res_lowq2_thr": res_lowq2_thr, "dis_even": dis_even, "norm": norm}


def response(item):
    """The weight function w(value, events) of a dial of the configuration."""
    if item["response"] not in RESPONSES:
        raise ValueError(f"unknown stress-test response {item['response']}")
    return RESPONSES[item["response"]]


def compute(ev, item):
    """Knots, weights (n_events, n_knots) of one stress-test dial, and the events it acts on: those of its modes whose weight differs
    from 1 at some knot."""
    knots = np.asarray(item["knots"], float)
    W = response(item)(knots[None, :], {k: v[:, None] for k, v in ev.items()})
    applies = np.isin(ev["Mode"], item["modes"]) & (np.abs(W - 1).max(1) > 1e-12)
    return knots, W, applies


def truth_weights(ev, items):
    """Weight of each event with every stress-test dial at its `truth`: the pseudo-data's weights."""
    w = np.ones(len(ev["Mode"]))
    for item in items:
        sel = np.isin(ev["Mode"], item["modes"])
        w[sel] *= response(item)(item["truth"], {k: v[sel] for k, v in ev.items()})
    return w
