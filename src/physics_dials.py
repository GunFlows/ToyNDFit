"""Physics dials: per-event weights from published formulas, computed from the NUISANCE flat tree (no generator call).

Each dial gives, for every event, its weight at a list of dial values (knots) relative to the sample as generated; the producer turns
the rows into the same per-event graphs as the polynomial dials. A dial acts only on its events (`applies`); the others get no graph.

Weights that depend on the hadronic current are ratios of the lepton tensor contracted with the hadron tensor, split into its vector,
axial and interference parts:
  CCQE: single-nucleon current at the event's struck and outgoing nucleon four-momenta (before FSI), de Forest's q~ = p_out - p_in,
        form factors at the lepton Q^2 (BBBA05 vector, hep-ex/0602017; dipole axial; PCAC pseudoscalar).
  Single pion: the Delta structure functions V1..V5 of Lalakulich & Paschos, PRD 71 074003 (hep-ph/0501109), App. A and Eq. (II.10).
Sample-level quantities (fate fractions, angular moments, multiplicities) are measured on the same events.
References are given per dial in the configuration (config/config_physics_dials.yaml) and above each dial in compute().

Created for the GuNFlows scaling study, October 2026.
"""
import os
import numpy as np
import awkward as ak
import uproot

M_P, M_N, M_AV, M_MU, M_PI, M_PI0 = 0.938272, 0.939565, 0.938919, 0.105658, 0.139570, 0.134977
MU_P, MU_N, G_A = 2.7928, -1.9130, 1.2723
NOMINAL = {"MA_QE": 1.21, "MA_RES": 0.95, "CA5": 1.01, "MV_RES": 0.84}   # NEUT defaults the sample is assumed to be generated with

KEYS = ["Mode", "PDGnu", "Enu_true", "tgt", "Q2", "q0", "q3", "W", "x", "CosThetaAdler", "PhiAdler", "ninitp",
        "pdg_init", "px_init", "py_init", "pz_init", "E_init", "pdg_vert", "px_vert", "py_vert", "pz_vert", "E_vert",
        "pdg", "px", "py", "pz", "E"]
CARBON = 1000060120


def load(file_path, n_events, tree_name="FlatTree_VARS"):
    """The first n_events entries of the flat tree, in file order (the producer's order)."""
    return uproot.open(file_path)[tree_name].arrays(KEYS, entry_stop=n_events)


# ---------------------------------------------------------------- Dirac algebra (Dirac representation, metric +,-,-,-)
_I2, _Z2 = np.eye(2), np.zeros((2, 2))
_SIG = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
GAM = np.array([np.block([[_I2, _Z2], [_Z2, -_I2]])] + [np.block([[_Z2, s], [-s, _Z2]]) for s in _SIG], dtype=complex)
G5 = np.block([[_Z2, _I2], [_I2, _Z2]]).astype(complex)
METRIC = np.array([1.0, -1.0, -1.0, -1.0])
ONE = np.eye(4, dtype=complex)


def slash(p):
    """p-slash for an array of four-vectors (n, 4) -> (n, 4, 4)."""
    return np.einsum("nm,mab->nab", p * METRIC, GAM)


def lepton_tensor(k, kp, antinu):
    """L^{mu nu} = Tr[A g^mu (1-g5) B g^nu (1-g5)], A = kp-slash, B = k-slash for neutrinos (swapped for antineutrinos).
    Reference: Llewellyn Smith, Phys. Rep. 3 (1972) 261."""
    gp = np.einsum("mab,bc->mac", GAM, ONE - G5)
    A, B = (slash(k), slash(kp)) if antinu else (slash(kp), slash(k))
    return np.einsum("nab,mbc,ncd,vda->nmv", A, gp, B, gp)


def nucleon_vertex(q, F1, F2, FA, FP):
    """Gamma^mu = F1 g^mu + i F2/(2M) sigma^{mu a} q_a + FA g^mu g5 + FP/M q^mu g5, for each event: (n, 4, 4, 4).
    Reference: Llewellyn Smith, Phys. Rep. 3 (1972) 261 (charged-current nucleon current)."""
    qs = slash(q)
    comm = np.einsum("mab,nbc->nmac", GAM, qs) - np.einsum("nab,mbc->nmac", qs, GAM)       # [g^mu, q-slash]
    g = (F1[:, None, None, None] * GAM[None] - (F2 / (4 * M_AV))[:, None, None, None] * comm
         + FA[:, None, None, None] * np.einsum("mab,bc->mac", GAM, G5)[None]
         + (FP / M_AV)[:, None, None, None] * q[:, :, None, None] * G5[None, None])
    return g


def hadron_tensor(p_in, m_in, p_out, m_out, gam):
    """H^{mu nu} = Tr[(p_out-slash + m_out) Gamma^mu (p_in-slash + m_in) Gamma-bar^nu], Gamma-bar = g0 Gamma^dagger g0.
    Reference: Llewellyn Smith, Phys. Rep. 3 (1972) 261."""
    A = slash(p_out) + m_out[:, None, None] * ONE
    B = slash(p_in) + m_in[:, None, None] * ONE
    gbar = np.einsum("ab,nmcb,cd->nmad", GAM[0], np.conj(gam), GAM[0])
    return np.einsum("nab,nmbc,ncd,nvda->nmv", A, gam, B, gbar)


def contract(L, H):
    """L_{mu nu} H^{mu nu} (real part; the imaginary part vanishes)."""
    return np.real(np.einsum("nmv,m,v,nmv->n", L, METRIC, METRIC, H))


# ---------------------------------------------------------------- nucleon form factors
def _bbba05(tau, a, b):
    """BBBA05 rational parametrisation of a Sachs form factor in tau = Q2/(4M^2) (Bradford et al., hep-ex/0602017, Table 1)."""
    num = sum(ak_ * tau ** k for k, ak_ in enumerate(a))
    den = 1 + sum(bk * tau ** (k + 1) for k, bk in enumerate(b))
    return num / den


def vector_ff(Q2, kind="BBBA05", mv=0.84):
    """Isovector F1, F2 from the Sachs form factors: BBBA05 (Bradford et al., hep-ex/0602017, Table 1) or dipole (G_En = 0)."""
    tau = Q2 / (4 * M_AV ** 2)
    if kind == "BBBA05":
        GEp = _bbba05(tau, [1, -0.0578], [11.1, 13.6, 33.0])
        GMp = MU_P * _bbba05(tau, [1, 0.150], [11.1, 19.6, 7.54])
        GEn = _bbba05(tau, [0, 1.25, 1.30], [-9.86, 305, -758, 802])
        GMn = MU_N * _bbba05(tau, [1, 1.81], [14.1, 20.7, 68.7])
    else:
        GD = 1 / (1 + Q2 / mv ** 2) ** 2
        GEp, GMp, GEn, GMn = GD, MU_P * GD, 0 * GD, MU_N * GD
    GE, GM = GEp - GEn, GMp - GMn
    return (GE + tau * GM) / (1 + tau), (GM - GE) / (1 + tau)


def axial_ff(Q2, ma):
    """Dipole axial form factor F_A = -g_A/(1 + Q2/M_A^2)^2 (g_A = 1.2723) and the pseudoscalar F_P from PCAC. Reference: Llewellyn Smith, Phys. Rep. 3 (1972) 261."""
    FA = -G_A / (1 + Q2 / ma ** 2) ** 2
    return FA, 2 * M_AV ** 2 * FA / (M_PI ** 2 + Q2)


def ccqe_contraction(k, kp, p_in, m_in, p_out, m_out, antinu, Q2, ma, vff="BBBA05", chunk=20000):
    """L.H for each event at axial mass ma and vector form factors vff.
    Reference: Llewellyn Smith, Phys. Rep. 3 (1972) 261."""
    out = np.empty(len(k))
    F1, F2 = vector_ff(Q2, vff)
    FA, FP = axial_ff(Q2, ma)
    for s in range(0, len(k), chunk):
        e = slice(s, s + chunk)
        q = p_out[e] - p_in[e]
        an = antinu[e][:, None, None]
        L = np.where(an, lepton_tensor(k[e], kp[e], True), lepton_tensor(k[e], kp[e], False))
        H = hadron_tensor(p_in[e], m_in[e], p_out[e], m_out[e], nucleon_vertex(q, F1[e], F2[e], FA[e], FP[e]))
        out[e] = contract(L, H)
    return out


def llewellyn_smith(E, Q2, ma, antinu, ml=M_MU, vff="BBBA05", fp_scale=1.0):
    """dsigma/dQ2 up to a constant, Formaggio & Zeller (arXiv:1305.7513) Eqs. 57-60: for the free-nucleon check."""
    M = M_AV
    F1, F2 = vector_ff(Q2, vff); FA, FP = axial_ff(Q2, ma)
    FP = FP * fp_scale
    eta = Q2 / (4 * M ** 2)
    A = (ml ** 2 + Q2) / M ** 2 * ((1 + eta) * FA ** 2 - (1 - eta) * F1 ** 2 + eta * (1 - eta) * F2 ** 2 + 4 * eta * F1 * F2
                                   - ml ** 2 / (4 * M ** 2) * ((F1 + F2) ** 2 + (FA + 2 * FP) ** 2 - (Q2 / M ** 2 + 4) * FP ** 2))
    B = Q2 / M ** 2 * FA * (F1 + F2)
    C = 0.25 * (FA ** 2 + F1 ** 2 + eta * F2 ** 2)
    su = 4 * M * E - Q2 - ml ** 2
    return (A + (1 if antinu else -1) * B * su / M ** 2 + C * su ** 2 / M ** 4) / E ** 2


# ---------------------------------------------------------------- event content
def _vec(a, pre, mask=None):
    """(n, 4) array of the first particle (per event) of a jagged list selected by mask; zeros where none."""
    E, px, py, pz = (a[f"E{pre}"], a[f"px{pre}"], a[f"py{pre}"], a[f"pz{pre}"])
    if mask is not None:
        E, px, py, pz = E[mask], px[mask], py[mask], pz[mask]
    f = lambda x: ak.to_numpy(ak.fill_none(ak.firsts(x), 0.0)).astype(float)
    return np.stack([f(E), f(px), f(py), f(pz)], 1)


class Events:
    """Per-event quantities used by the dials (numpy arrays over the loaded events)."""

    def __init__(self, a):
        self.a = a
        self.mode = ak.to_numpy(a.Mode).astype(int)
        self.am = np.abs(self.mode)
        self.antinu = self.mode < 0
        self.E = ak.to_numpy(a.Enu_true).astype(float)
        self.Q2 = ak.to_numpy(a.Q2).astype(float)
        self.q0 = ak.to_numpy(a.q0).astype(float)
        self.q3 = ak.to_numpy(a.q3).astype(float)
        self.W = ak.to_numpy(a.W).astype(float)
        self.carbon = ak.to_numpy(a.tgt) == CARBON
        self.out = ak.local_index(a.pdg_vert) >= a.ninitp          # vertex list = initial particles, then the outgoing ones
        self.pdg_out = a.pdg_vert[self.out]
        self.p_vert = np.sqrt(a.px_vert ** 2 + a.py_vert ** 2 + a.pz_vert ** 2)[self.out]
        self.p_fin = np.sqrt(a.px ** 2 + a.py ** 2 + a.pz ** 2)
        self.T_fin = a.E - np.sqrt(np.maximum(a.E ** 2 - self.p_fin ** 2, 0))
        isnuc = lambda x: (x == 2212) | (x == 2112)
        ispi = lambda x: (abs(x) == 211) | (x == 111)
        self.nuc_out, self.pi_out = isnuc(self.pdg_out), ispi(self.pdg_out)
        self.nuc_fin, self.pi_fin = isnuc(a.pdg), ispi(a.pdg)
        self.n_nuc_out = ak.to_numpy(ak.sum(self.nuc_out, 1)); self.n_nuc_fin = ak.to_numpy(ak.sum(self.nuc_fin, 1))
        self.n_pi_out = ak.to_numpy(ak.sum(self.pi_out, 1)); self.n_pi_fin = ak.to_numpy(ak.sum(self.pi_fin, 1))
        lep = (abs(self.pdg_out) == 11) | (abs(self.pdg_out) == 13)
        self.other_out = ak.to_numpy(ak.sum(~(self.nuc_out | self.pi_out | lep) & (abs(self.pdg_out) < 1000000000), 1))
        inuc = isnuc(a.pdg_init)
        self.init_nucs = a.pdg_init[inuc]

    def same_momenta(self, sel_out, sel_fin):
        """True where the selected outgoing vertex particles and final particles agree (same count, same sorted momenta)."""
        a = ak.sort(self.p_vert[sel_out], axis=1); b = ak.sort(self.p_fin[sel_fin], axis=1)
        same_n = ak.to_numpy(ak.num(a)) == ak.to_numpy(ak.num(b))
        ok = np.zeros(len(same_n), bool)
        idx = np.where(same_n)[0]
        if len(idx):
            d = ak.to_numpy(ak.fill_none(ak.max(abs(a[idx] - b[idx]), 1), 0.0))
            ok[idx] = d < 1e-4
        return ok


def ccqe_weights(ev, knots, what):
    """CCQE weights at the knots: what = 'MA' (axial mass) or 'VFF' (x = 0 BBBA05, 1 dipole vector form factors; cross section linear in x)."""
    s = ev.am == 1
    a = ev.a[s]
    k = _vec(a, "_init", abs(a.pdg_init) == 14)
    outi = ak.local_index(a.pdg_vert) >= a.ninitp
    kp = _vec(a, "_vert", (abs(a.pdg_vert) == 13) & outi)
    p_in = _vec(a, "_init", (a.pdg_init == 2112) | (a.pdg_init == 2212))
    p_out = _vec(a, "_vert", ((a.pdg_vert == 2112) | (a.pdg_vert == 2212)) & outi)
    anti = ev.antinu[s]
    m_in = np.where(anti, M_P, M_N); m_out = np.where(anti, M_N, M_P)
    Q2 = ev.Q2[s]
    nom = ccqe_contraction(k, kp, p_in, m_in, p_out, m_out, anti, Q2, NOMINAL["MA_QE"])
    W = np.ones((len(ev.mode), len(knots)))
    if what == "MA":
        for j, x in enumerate(knots):
            W[s, j] = ccqe_contraction(k, kp, p_in, m_in, p_out, m_out, anti, Q2, x) / nom
    else:
        dip = ccqe_contraction(k, kp, p_in, m_in, p_out, m_out, anti, Q2, NOMINAL["MA_QE"], vff="dipole") / nom
        for j, x in enumerate(knots):
            W[s, j] = 1 - x + x * dip
    return np.maximum(W, 0), s


# ---------------------------------------------------------------- Delta(1232): Lalakulich & Paschos, hep-ph/0501109
def delta_cross_section(E, Q2, W, r, antinu):
    """Eq. (II.10) bracket with the V_i of App. A at the event's (E, Q2, W), nucleon at rest, M_R -> W; every axial form factor
    multiplied by r (all proportional to C5A: C4A = -C5A/4, C3A = 0, C6A = C5A m_N^2/(Q^2 + m_pi^2), Eq. II.11).
    Reference: Lalakulich & Paschos, PRD 71 074003 (2005), hep-ph/0501109."""
    mN, mR, ml = M_AV, W, M_MU
    MV = NOMINAL["MV_RES"]
    C3V = 1.95 / (1 + Q2 / MV ** 2) ** 2 / (1 + Q2 / (4 * MV ** 2))
    C4V = -C3V * mN / W
    C5A = r * NOMINAL["CA5"] / (1 + Q2 / NOMINAL["MA_RES"] ** 2) ** 2
    C4A = -C5A / 4; C6A = C5A * mN ** 2 / (Q2 + M_PI ** 2)
    nu = (W ** 2 + Q2 - mN ** 2) / (2 * mN); qp = mN * nu; kp_ = mN * E; kpp = mN * (E - nu)
    a = qp - Q2
    V1 = 3 * (C3V ** 2 / mN ** 2 * 2 / (3 * mR ** 2) * (a ** 2 * (qp + mN ** 2) + mR ** 2 * (qp ** 2 + Q2 * mN ** 2 + Q2 * mN * mR))
              + C4V ** 2 / mN ** 4 * 2 / 3 * a ** 2 * (qp + mN ** 2 - mN * mR)
              + C3V * C4V / mN ** 3 * 2 / (3 * mR) * a * (a * (qp + mN ** 2 - 2 * mN * mR) + mR ** 2 * qp)
              + 2 / 3 * ((C4A / mN ** 2) ** 2 * a ** 2 + C5A ** 2 + 2 * C4A * C5A / mN ** 2 * a) * (qp + mN ** 2 + mN * mR))
    V2 = 3 * (C3V ** 2 * 2 / (3 * mR ** 2) * Q2 * (qp + mN ** 2 + mR ** 2) + C4V ** 2 / mN ** 2 * 2 / 3 * Q2 * (qp + mN ** 2 - mN * mR)
              + C3V * C4V / mN * 2 / (3 * mR) * Q2 * (qp + (mR - mN) ** 2)
              + 2 / 3 * (C5A ** 2 * mN ** 2 / mR ** 2 + C4A ** 2 / mN ** 2 * Q2) * (qp + mN ** 2 + mN * mR))
    V3 = 3 * (4 / (3 * mR) * (-C3V * C4A / mN * a - C3V * C5A * mN) * (2 * mR ** 2 + 2 * mN * mR + Q2 - qp)
              + 4 / 3 * a * (-C4V * C4A / mN ** 2 * a - C4V * C5A))
    V4 = 3 * (2 / (3 * mR ** 2) * C3V ** 2 * ((2 * qp - Q2) * (qp + mN ** 2) - mR ** 2 * (mN ** 2 + mN * mR))
              + 2 / 3 * C4V ** 2 / mN ** 2 * (2 * qp - Q2) * (qp + mN ** 2 - mN * mR)
              + 2 / (3 * mR) * C3V * C4V / mN * ((2 * qp - Q2) * (qp + mN ** 2 - 2 * mN * mR) + qp * mR ** 2)
              + 2 / 3 * (C5A ** 2 * mN ** 2 / mR ** 2 + C4A ** 2 / mN ** 2 * (2 * qp - Q2) + C6A ** 2 / (mN ** 2 * mR ** 2) * ((Q2 - qp) ** 2 + Q2 * mR ** 2)
                         + 2 * C4A * C5A - 2 * C4A * C6A / mN ** 2 * qp - 2 * C5A * C6A / mR ** 2 * (mR ** 2 + Q2 - qp)) * (qp + mN ** 2 + mN * mR))
    V5 = 3 * (2 / 3 * C3V ** 2 / mR ** 2 * qp * (qp + mN ** 2 + mR ** 2) + 2 / 3 * C4V ** 2 / mN ** 2 * qp * (qp + mN ** 2 - mN * mR)
              + 2 / (3 * mR) * C3V * C4V / mN * qp * (qp + (mR - mN) ** 2)
              + 2 / 3 * (C4A ** 2 / mN ** 2 * qp + C5A ** 2 * mN ** 2 / mR ** 2 + C4A * C5A - C4A * C6A / mN ** 2 * Q2
                         + C5A * C6A / mR ** 2 * (qp - Q2)) * (qp + mN ** 2 + mN * mR))
    s3 = np.where(antinu, 1.0, -1.0)
    return (V1 * (Q2 + ml ** 2) + V2 / mN ** 2 * (2 * kp_ * kpp - 0.5 * mN ** 2 * (Q2 + ml ** 2))
            + s3 * V3 / mN ** 2 * (Q2 * kp_ - 0.5 * qp * (Q2 + ml ** 2))
            + V4 / mN ** 2 * ml ** 2 * (Q2 + ml ** 2) / 2 - 2 * V5 / mN ** 2 * ml ** 2 * kp_)


def res_axial_weights(ev, knots, what):
    """MA_RES or CA5: weight = sigma(r)/sigma(1), r = C5A'(Q2)/C5A(Q2); sigma is quadratic in r (vector, interference, axial parts).
    Reference: Lalakulich & Paschos, PRD 71 074003 (2005), hep-ph/0501109."""
    s = np.isin(ev.am, (11, 12, 13)) & np.isfinite(ev.W) & (ev.Q2 >= 0)
    E, Q2, W, an = ev.E[s], ev.Q2[s], np.maximum(ev.W[s], M_AV + M_PI), ev.antinu[s]
    S0 = delta_cross_section(E, Q2, W, 0.0, an); Sp = delta_cross_section(E, Q2, W, 1.0, an); Sm = delta_cross_section(E, Q2, W, -1.0, an)
    b = (Sp - Sm) / 2; c = (Sp + Sm) / 2 - S0
    out = np.ones((len(ev.mode), len(knots)))
    for j, x in enumerate(knots):
        r = (x / NOMINAL["CA5"]) * np.ones_like(Q2) if what == "CA5" else ((1 + Q2 / NOMINAL["MA_RES"] ** 2) / (1 + Q2 / x ** 2)) ** 2
        out[s, j] = np.maximum(S0 + b * r + c * r ** 2, 0) / Sp
    return out, s


# ---------------------------------------------------------------- simple event classes
def linear_class_weights(sel, knots, nominal=1.0):
    """Normalisation of a class of events: w = x / nominal."""
    W = np.ones((len(sel), len(knots)))
    W[sel] = np.asarray(knots, float)[None, :] / nominal
    return W, sel


def balanced_weights(up, down, frac_up, knots):
    """One class scaled by (1 + x), its complement by 1 - x f/(1 - f), f = the class's share among up | down: total unchanged."""
    W = np.ones((len(up), len(knots)))
    x = np.asarray(knots, float)[None, :]
    W[up] = 1 + x
    W[down] = np.maximum(1 - x * (frac_up[down] / np.maximum(1 - frac_up[down], 1e-6))[:, None], 0)
    return np.maximum(W, 0), up | down


def gauss2d(q0, q3, N, m0, m3, s0, s3, rho):
    """Correlated Gaussian in (q0, q3) of the MINERvA low-recoil tune (arXiv:1803.09377), parameters MINERVA_GAUSS."""
    z = (q0 - m0) ** 2 / s0 ** 2 + (q3 - m3) ** 2 / s3 ** 2 - 2 * rho * (q0 - m0) * (q3 - m3) / (s0 * s3)
    return N * np.exp(-0.5 * z / (1 - rho ** 2))


# MINERvA low-recoil Gaussians (N, mu_q0, mu_q3, sigma_q0, sigma_q3, rho), as published in NuSystematics/nusystematics
# (responsecalculators/MINERvAq0q3Weighting_data.hh): MnvTune v1 fits, arXiv:1803.09377; weight 1 + G.
MINERVA_GAUSS = {"2p2h_all": (10.5798, 0.254032, 0.50834, 0.0571035, 0.129051, 0.875287),
                 "2p2h_nn": (17.0344, 0.289916, 0.532062, 0.0746849, 0.137321, 0.836689),
                 "2p2h_np": (8.58724, 0.23626, 0.502603, 0.072291, 0.154832, 0.789796),
                 "qe": (5.38719, 0.213611, 0.396552, 0.0496312, 0.125062, 0.806659)}


def minerva_gauss_weights(ev, sel, which, knots):
    """w = 1 + x G(q0, q3), MINERvA low-recoil tune (MnvTune v1, arXiv:1803.09377; parameters as in NuSystematics)."""
    G = gauss2d(ev.q0[sel], ev.q3[sel], *MINERVA_GAUSS[which])
    W = np.ones((len(ev.mode), len(knots)))
    W[sel] = 1 + np.asarray(knots, float)[None, :] * G[:, None]
    return W, sel


def pair_type(ev):
    """2p2h initial pair: True for a proton-neutron pair."""
    n_p = ak.to_numpy(ak.sum(ev.init_nucs == 2212, 1)); n_n = ak.to_numpy(ak.sum(ev.init_nucs == 2112, 1))
    return (n_p == 1) & (n_n == 1)


def share_by_bins(cls, pop, var, edges):
    """Share of `cls` events among `pop` events in bins of var, returned per event (pop events only meaningful)."""
    b = np.clip(np.digitize(var, edges) - 1, 0, len(edges) - 2)
    num = np.bincount(b[pop & cls], minlength=len(edges) - 1).astype(float)
    den = np.bincount(b[pop], minlength=len(edges) - 1).astype(float)
    return (num / np.maximum(den, 1))[b]


# ---------------------------------------------------------------- Delta decay: Rein-Sehgal against isotropic in the Adler frame
def delta_decay_weights(ev, knots):
    """x = 0: as generated (Rein-Sehgal Delta decay), 1: isotropic. The generated angular density in each (mode, W, Q2) bin is the
    Rein-Sehgal form W = [1 + a(3c^2 - 1) + b sin2t cos(phi) + d sin^2t cos(2phi)]/4pi (Rein & Sehgal 1981, Eq. 2.46), its
    coefficients measured on the sample from orthogonal moments: a = <3c^2-1>/(4/5), b = <sin2t cos phi>/(4/15), d = <sin^2t cos2phi>/(4/15)."""
    c = ak.to_numpy(ev.a.CosThetaAdler).astype(float); ph = np.radians(ak.to_numpy(ev.a.PhiAdler).astype(float))
    s = np.isin(ev.am, (11, 12, 13)) & (c > -1.5) & np.isfinite(ev.W) & (ev.Q2 >= 0)
    st = np.sqrt(np.clip(1 - c ** 2, 0, 1))
    y1, y2, y3 = 3 * c ** 2 - 1, 2 * st * c * np.cos(ph), st ** 2 * np.cos(2 * ph)
    wb = np.digitize(ev.W, [1.2, 1.3, 1.4, 1.6, 2.0]); qb = np.digitize(ev.Q2, [0.1, 0.25, 0.5, 1.0])
    key = ev.mode * 100 + wb * 10 + qb
    dens = np.ones(len(c))
    for kk in np.unique(key[s]):
        m = s & (key == kk)
        a_, b_, d_ = y1[m].mean() / 0.8, y2[m].mean() / (4 / 15), y3[m].mean() / (4 / 15)
        dens[m] = 1 + a_ * y1[m] + b_ * y2[m] + d_ * y3[m]
    w_iso = np.clip(1 / np.maximum(dens, 0.2), 0, 5)
    W = np.ones((len(c), len(knots)))
    x = np.asarray(knots, float)[None, :]
    W[s] = 1 - x + x * w_iso[s, None]
    return W, s


# ---------------------------------------------------------------- low-Q2 single-pion suppression (MINERvA PRD 100 072005, Eqs. 6-7)
LOWQ2_CHANNELS = {   # MINERvA PRD 100 072005 Table IX (FrAbs + low-Q2 fit): R1, R2 per channel; resonant events on carbon
    "nu_1pic": (0.53, 0.50), "nu_Npic": (0.43, 0.50), "nu_1pi0": (0.21, 0.63), "nubar_1pi0": (0.14, 1.00)}


def lowq2_channel(ev, channel):
    """Resonant events of a MINERvA channel: nu 1 charged pion (modes 11, 13, at most one final pion), nu N pions (any resonant nu event
    with two or more final pions), nu 1 neutral pion (mode 12, at most one final pion), nubar (every resonant nubar event: the only
    antineutrino channel MINERvA fitted, CC1pi0, applied to all of them).
    Reference: MINERvA PRD 100 072005 (arXiv:1903.01558) Table IX."""
    res = np.isin(ev.am, (11, 12, 13)); nu = ~ev.antinu; one = ev.n_pi_fin <= 1
    return {"nu_1pic": res & nu & np.isin(ev.am, (11, 13)) & one, "nu_Npic": res & nu & ~one,
            "nu_1pi0": res & nu & (ev.am == 12) & one, "nubar_1pi0": res & ev.antinu}[channel]


def lowq2_pion_weights(ev, knots, R1=0.32, R2=0.5, sel=None):
    """w = 1 - x (1 - R1)(1 - R(Q2))^2 for Q2 < 0.7 GeV^2, R a quadratic through (0, 0), (0.35, R2), (0.7, 1); x = 1: the fitted
    suppression of the channel (MINERvA PRD 100 072005 Eqs. 6-7), resonant events on carbon; the dial form follows nusystematics."""
    s = (np.isin(ev.am, (11, 12, 13)) if sel is None else sel) & ev.carbon & (ev.Q2 >= 0)
    q = ev.Q2
    x1, x2, x3 = 0.0, 0.35, 0.7
    R = R2 * (q - x1) * (q - x3) / ((x2 - x1) * (x2 - x3)) + (q - x1) * (q - x2) / ((x3 - x1) * (x3 - x2))
    supp = np.where(q < x3, (1 - R1) * (1 - R) ** 2, 0.0)
    W = np.ones((len(q), len(knots)))
    W[s] = 1 - np.asarray(knots, float)[None, :] * supp[s, None]
    return np.maximum(W, 0), s


# ---------------------------------------------------------------- multi-pion multiplicity: NEUT -> AGKY (Yang et al., EPJC 63 1, Table 1)
AGKY = {(False, 2212): (0.40, 1.42, 7.93), (False, 2112): (-0.20, 1.42, 5.22), (True, 2212): (0.02, 1.28, 5.22), (True, 2112): (0.80, 0.95, 7.93)}


def agky_prob(npi, W, a, b, c):
    """KNO-Levy probability of n = npi + 1 hadrons, <n> = 1.5 (a + b ln W^2), normalised over npi = 2..7 (NEUT's range).
    Reference: AGKY, arXiv:0904.4043."""
    from scipy.special import gammaln
    def f(n, nav):
        z = n / nav; return np.exp(np.log(2) - c + (c * z + 1) * np.log(c) - gammaln(c * z + 1)) / nav
    nav = np.maximum(1.5 * (a + b * np.log(W ** 2)), 0.5)   # W below the AGKY range (NUISANCE's W includes Fermi motion)
    tot = sum(f(k + 1, nav) for k in range(2, 8))
    return f(npi + 1, nav) / tot


def multiplicity_weights(ev, knots):
    """Mode +-21 (1.3 < W < 2 GeV): w = 1 + x (P_AGKY(n|W) / P_sample(n|W) - 1); P_sample measured in 50 MeV bins of W per channel.
    Reference: AGKY, arXiv:0904.4043; T2K arXiv:2606.14015."""
    s = (ev.am == 21) & (ev.n_pi_out >= 2) & np.isfinite(ev.W)
    npi = np.minimum(ev.n_pi_out, 7)
    struck = ak.to_numpy(ak.fill_none(ak.firsts(ev.init_nucs), 0))
    W = np.ones((len(npi), len(knots)))
    edges = np.arange(1.3, 2.05, 0.05)
    wb = np.clip(np.digitize(ev.W, edges) - 1, 0, len(edges) - 2)
    ratio = np.ones(len(npi))
    for anti in (False, True):
        for nuc in (2212, 2112):
            m = s & (ev.antinu == anti) & (struck == nuc)
            if not m.any(): continue
            a, b, c = AGKY[(anti, nuc)]
            cnt = np.zeros((len(edges) - 1, 8))
            np.add.at(cnt, (wb[m], npi[m]), 1)
            ps = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
            psam = ps[wb[m], npi[m]]
            ratio[m] = agky_prob(npi[m], ev.W[m], a, b, c) / np.maximum(psam, 1e-6)
    W[s] = np.maximum(1 + np.asarray(knots, float)[None, :] * (ratio[s, None] - 1), 0)
    return W, s


# ---------------------------------------------------------------- pion final-state interactions: fates from each event's history
def pion_fates(ev):
    """Event-level pion fate on carbon, for events whose outgoing vertex hadrons are nucleons and pions only:
    0 none, 1 untouched, 2 quasi-elastic scatter, 3 charge exchange, 4 absorption, 5 production; and the leading vertex pion momentum.
    Reference: T2K arXiv:2303.03222 and arXiv:2606.14015 (pion FSI dials); fate classes as GENIE's (arXiv:1510.05494)."""
    ok = ev.carbon & (ev.n_pi_out > 0) & (ev.other_out == 0)
    p_lead = ak.to_numpy(ak.fill_none(ak.max(ev.p_vert[ev.pi_out], 1), 0.0))
    ch = lambda pdg, m: (ak.to_numpy(ak.sum((pdg == 211) & m, 1)), ak.to_numpy(ak.sum((pdg == 111) & m, 1)), ak.to_numpy(ak.sum((pdg == -211) & m, 1)))
    co, cf = ch(ev.pdg_out, ev.pi_out), ch(ev.a.pdg, ev.pi_fin)
    same_q = (co[0] == cf[0]) & (co[1] == cf[1]) & (co[2] == cf[2])
    untouched = ev.same_momenta(ev.pi_out, ev.pi_fin) & same_q
    fate = np.zeros(len(ok), int)
    fate[ok & (ev.n_pi_fin > ev.n_pi_out)] = 5
    fate[ok & (ev.n_pi_fin < ev.n_pi_out)] = 4
    eq = ok & (ev.n_pi_fin == ev.n_pi_out)
    fate[eq & ~same_q] = 3
    fate[eq & same_q] = 2
    fate[eq & untouched] = 1
    return fate, p_lead


PION_FSI = {"QE_low": (2, "low"), "QE_high": (2, "high"), "CEX_low": (3, "low"), "CEX_high": (3, "high"), "ABS": (4, "all"), "PROD": (5, "all")}


def pion_fsi_weights(ev, which, knots, fate=None, p_lead=None):
    """Fate `which` scaled by (1 + x) in its momentum range (low: leading vertex pion below 0.5 GeV/c, high: above), all the other fates
    of the same momentum (untouched included) taking up the difference in proportion to their shares, so the total is unchanged
    (GENIE hA fates reweighting with a cushion, arXiv:1510.05494 Eqs. 9.16-9.18); shares measured on the sample in 50 MeV/c bins of the
    leading vertex pion momentum, per beam.
    Reference: T2K arXiv:2303.03222 and arXiv:2606.14015 (pion FSI dials); fate classes as GENIE's (arXiv:1510.05494)."""
    if fate is None: fate, p_lead = pion_fates(ev)
    c, rng = PION_FSI[which]
    inr = np.ones(len(fate), bool) if rng == "all" else ((p_lead < 0.5) if rng == "low" else (p_lead >= 0.5))
    edges = np.arange(0, 1.55, 0.05)
    pop = fate > 0
    W = np.ones((len(fate), len(knots)))
    x = np.asarray(knots, float)[None, :]
    for anti in (False, True):
        P = pop & (ev.antinu == anti)
        fc = share_by_bins(fate == c, P, p_lead, edges)
        up = P & (fate == c) & inr; cu = P & (fate != c) & inr
        W[up] = 1 + x
        W[cu] = 1 - x * (fc[cu] / np.maximum(1 - fc[cu], 1e-6))[:, None]
    sel = pop & inr
    return np.maximum(W, 0), sel


# ---------------------------------------------------------------- nucleon final-state interactions (T2K 2606.14015 Sec. V B 5; GENIE fates)
def nucleon_fsi_classes(ev):
    """QE and 2p2h events on carbon: fsi = the final nucleons differ from the outgoing vertex nucleons; knock-out = more nucleons
    (or pions) in the final state than at the vertex; elastic = the other re-interacting events.
    Reference: T2K arXiv:2606.14015 Sec. V B 5; GENIE's fates (arXiv:1510.05494)."""
    pop = np.isin(ev.am, (1, 2)) & ev.carbon
    same = ev.same_momenta(ev.nuc_out, ev.nuc_fin)
    fsi = pop & ~same
    ko = fsi & ((ev.n_nuc_fin > ev.n_nuc_out) | (ev.n_pi_fin > 0))
    return pop, fsi, ko


def nucleon_fsi_weights(ev, which, knots):
    """which = 'total': re-interacting QE/2p2h events up, the others down, inclusive fixed per mode and beam; otherwise knock-out against elastic, re-interacting total fixed. Reference: T2K arXiv:2606.14015 Sec. V B 5."""
    pop, fsi, ko = nucleon_fsi_classes(ev)
    if which == "total":     # T2K's nucleon FSI: re-interacting events up, the others down, inclusive fixed per mode and beam
        up, down = fsi, pop & ~fsi
    else:                    # knock-out against elastic re-interaction, the re-interacting total fixed
        up, down = ko, fsi & ~ko
    f = np.zeros(len(pop))
    for md in (1, 2):
        for anti in (False, True):
            m = (up | down) & (ev.am == md) & (ev.antinu == anti)
            if m.any(): f[m] = up[m].mean()
    return balanced_weights(up, down, f, knots)


# ---------------------------------------------------------------- DIS: GRV98 LO with and without the Bodek-Yang corrections
class GRV98:
    """GRV98 LO parton densities from the PDFLIB grid (as distributed with GENIE: data/evgen/pdfs/GRV98lo_patched.LHgrid): x times
    u_v, d_v, (d-bar - u-bar), (u-bar + d-bar), s, g on 67 x 27 points in (x, Q2), interpolated bilinearly in (ln x, ln Q2) after
    dividing out the small- and large-x behaviour, as grv98.f does; Q2 frozen at the grid's lowest value, 0.8 GeV^2.
    Reference: Glück, Reya & Vogt, Eur. Phys. J. C 5 (1998) 461."""
    POW = [(3, 0.5), (4, 0.5), (7, 0.5), (7, -0.2), (7, -0.2), (5, -0.2)]   # ((1-x) power, x power) divided out per column

    def __init__(self, path):
        lines = open(path).read().splitlines()
        i0 = next(i for i, l in enumerate(lines) if l.strip() == "'GRV'") + 2     # the x and Q2 grids follow the 'GRV' and '0,0' lines
        nums, i = [], i0
        while len(nums) < 68 + 27:
            nums += [float(v) for v in lines[i].split()]; i += 1
        self.x = np.array(nums[:68]); self.q2 = np.array(nums[68:95])
        i = next(j for j, l in enumerate(lines) if l.startswith("file:")) + 1
        rows = [list(map(float, lines[j].split())) for j in range(i, i + 67 * 27)]
        g = np.array(rows).reshape(67, 27, 6)
        xs = self.x[:67, None, None]
        red = np.empty_like(g)
        for c, (n1, nx) in enumerate(self.POW):
            red[..., c] = g[..., c] / ((1 - xs[..., 0]) ** n1 * xs[..., 0] ** nx)
        self.red = red; self.lx = np.log(self.x[:67]); self.lq = np.log(self.q2)

    def __call__(self, x, Q2):
        """(n, 6): x u_v, x d_v, x (d-bar - u-bar), x (u-bar + d-bar), x s, x g."""
        x = np.clip(x, self.x[0], self.x[66]); Q2 = np.clip(Q2, self.q2[0], self.q2[-1])
        lx, lq = np.log(x), np.log(Q2)
        i = np.clip(np.searchsorted(self.lx, lx) - 1, 0, 65); j = np.clip(np.searchsorted(self.lq, lq) - 1, 0, 25)
        tx = (lx - self.lx[i]) / (self.lx[i + 1] - self.lx[i]); tq = (lq - self.lq[j]) / (self.lq[j + 1] - self.lq[j])
        r = ((1 - tx) * (1 - tq))[:, None] * self.red[i, j] + (tx * (1 - tq))[:, None] * self.red[i + 1, j] \
            + ((1 - tx) * tq)[:, None] * self.red[i, j + 1] + (tx * tq)[:, None] * self.red[i + 1, j + 1]
        out = np.empty_like(r)
        for c, (n1, nx) in enumerate(self.POW):
            out[:, c] = r[:, c] * (1 - x) ** n1 * x ** nx
        return out


# Bodek-Yang "BY05" (NuInt04/DIS05: hep-ph/0411202, hep-ph/0508007; NEUT's default and GENIE BY00's parameters)
BY05 = dict(A=0.538, B=0.305, Cv1u=0.291, Cv2u=0.189, Cv1d=0.202, Cv2d=0.255, Csea_u=0.363, Csea_d=0.621, fPDF=1.015)


def r_world(x, Q2):
    """R = sigma_L / sigma_T, Whitlow et al. as used with BY05 (hep-ex/0308007 Eqs. 3-4): frozen at Q2 = 0.35, scaled below."""
    def rw(x, q):
        th = 1 + 12 * q / (q + 1) * 0.125 ** 2 / (0.125 ** 2 + x ** 2)
        return 0.0635 / np.log(q / 0.04) * th + 0.5747 / q - 0.3534 / (q ** 2 + 0.09)
    q = np.maximum(Q2, 1e-6)
    return np.where(q >= 0.35, rw(x, np.maximum(q, 0.35)), 3.207 * q / (q ** 2 + 1) * rw(x, 0.35))


def dis_structure(pdf, x, Q2, struck_p, antinu, by, kfac="vector"):
    """F2 and xF3 (per nucleon) of nu/nubar on a proton or neutron at Bjorken x, Q2: GRV98 at x (by = False), or with the BY05
    scaling variable xi_w, K factors on valence and sea, the d/u correction and the normalisation (by = True).
    Reference: Bodek & Yang, hep-ex/0308007 (BY05)."""
    M = M_AV
    if by:
        p = BY05
        xi = 2 * x * (Q2 + p["B"]) / (Q2 * (1 + np.sqrt(1 + 4 * M ** 2 * x ** 2 / np.maximum(Q2, 1e-9))) + 2 * p["A"] * x)
        f = pdf(np.clip(xi, 1e-9, 0.999), np.maximum(Q2, 0.8)) * p["fPDF"]
        GD2 = 1 / (1 + Q2 / 0.71) ** 4
        if kfac == "vector":
            Kvu = (1 - GD2) * (Q2 + p["Cv2u"]) / (Q2 + p["Cv1u"]); Kvd = (1 - GD2) * (Q2 + p["Cv2d"]) / (Q2 + p["Cv1d"])
            Ksu = Q2 / (Q2 + p["Csea_u"]); Ksd = Q2 / (Q2 + p["Csea_d"])
        else:   # axial K factors, Bodek-Yang 2021 (arXiv:2108.09240 Eqs. 47-51, Type II): the same for u and d
            Kvu = Kvd = (Q2 + 0.3 * 0.18) / (Q2 + 0.18)
            Ksu = Ksd = (Q2 + 0.55 * 0.75) / (Q2 + 0.75)
        uv, dv = f[:, 0], f[:, 1]
        dlt = -0.00817 + 0.0506 * xi + 0.0798 * xi ** 2                        # d/u correction (hep-ex/0308007 Eqs. 7-9)
        den = 1 + dlt * uv / np.maximum(uv + dv, 1e-12)
        uv, dv = uv / den, (dv + uv * dlt) / den
    else:
        f = pdf(np.clip(x, 1e-9, 0.999), np.maximum(Q2, 0.8))
        uv, dv = f[:, 0], f[:, 1]
        Kvu = Kvd = Ksu = Ksd = 1.0
    ub, db, s = (f[:, 3] - f[:, 2]) / 2, (f[:, 3] + f[:, 2]) / 2, f[:, 4]
    # proton: u = uv + ubar, d = dv + dbar; neutron: isospin rotation u <-> d
    U_v = np.where(struck_p, uv, dv); D_v = np.where(struck_p, dv, uv)
    Ub = np.where(struck_p, ub, db); Db = np.where(struck_p, db, ub)
    Kv_U = np.where(struck_p, Kvu, Kvd) if by else 1.0; Kv_D = np.where(struck_p, Kvd, Kvu) if by else 1.0
    Ks_U = np.where(struck_p, Ksu, Ksd) if by else 1.0; Ks_D = np.where(struck_p, Ksd, Ksu) if by else 1.0
    xu = Kv_U * U_v + Ks_U * Ub; xd = Kv_D * D_v + Ks_D * Db
    xub, xdb, xs = Ks_U * Ub, Ks_D * Db, (Ksd if by else 1.0) * s
    q_nu, qb_nu = xd + xs, xub          # nu: d, s quarks; u-bar antiquarks
    q_nb, qb_nb = xu, xdb + xs          # nubar: u quarks; d-bar, s-bar antiquarks
    q = np.where(antinu, q_nb, q_nu); qb = np.where(antinu, qb_nb, qb_nu)
    return 2 * (q + qb), 2 * (q - qb)


def dis_cross_section(pdf, ev, sel, by, dR=0.0, split=None):
    """d sigma / dx dy up to a constant, with the lepton mass (Albright & Jarlskog, Nucl. Phys. B84 (1975) 467; F4 = 0, x F5 = F2):
    (y^2 + y m^2/(2EMx)) xF1 + [1 - m^2/(4E^2) - (1 + Mx/(2E)) y - m^2/(EMx)] F2 +- [y (1 - y/2) - y m^2/(4EMx)] xF3,
    2xF1 = F2 (1 + 4M^2x^2/Q2)/(1 + R)."""
    a = ev.a[sel]
    x = ak.to_numpy(a.x).astype(float) if "x" in a.fields else ev.Q2[sel] / (2 * M_AV * ev.q0[sel])
    E, Q2, an = ev.E[sel], ev.Q2[sel], ev.antinu[sel]
    y = ev.q0[sel] / E
    struck_p = ak.to_numpy(ak.fill_none(ak.firsts(ev.init_nucs[sel]), 2212)) == 2212
    if split is None:
        F2, xF3 = dis_structure(pdf, x, Q2, struck_p, an, by)
    else:   # vector and axial halves (F2 = F2V + F2A, xF3 from the vector part: arXiv:1011.6592, 2108.09240), each BY on or off
        vec_on, ax_on = split
        F2V, xF3 = dis_structure(pdf, x, Q2, struck_p, an, vec_on, "vector")
        F2A, _ = dis_structure(pdf, x, Q2, struck_p, an, ax_on, "axial")
        F2 = 0.5 * (F2V + F2A)
    R = r_world(x, Q2) + dR
    xF1 = 0.5 * F2 * (1 + 4 * M_AV ** 2 * x ** 2 / np.maximum(Q2, 1e-9)) / (1 + R)
    M, m = M_AV, M_MU
    xs = np.maximum(x, 1e-6)
    return ((y ** 2 + y * m ** 2 / (2 * E * M * xs)) * xF1 + (1 - m ** 2 / (4 * E ** 2) - (1 + M * x / (2 * E)) * y - m ** 2 / (E * M * xs)) * F2
            + np.where(an, -1, 1) * (y * (1 - y / 2) - y * m ** 2 / (4 * E * M * xs)) * xF3)


_PDF = {}


def bodek_yang_weights(ev, knots, modes, grid):
    """x = 0: as generated (BY05 on), 1: BY corrections off (GRV98 at Bjorken x); linear in x, as T2K's on/off dials [C]."""
    if grid not in _PDF: _PDF[grid] = GRV98(grid)
    s = np.isin(ev.am, modes) & (ev.Q2 > 0) & (ev.q0 > 0)
    on = dis_cross_section(_PDF[grid], ev, s, True); off = np.maximum(dis_cross_section(_PDF[grid], ev, s, False), 0)
    good = on > 1e-12 * np.abs(on).max(); s[np.where(s)[0][~good]] = False; on, off = on[good], off[good]
    ratio = np.where(on > 0, off / np.maximum(on, 1e-30), 1.0)
    W = np.ones((len(ev.mode), len(knots)))
    W[s] = np.maximum(1 + np.asarray(knots, float)[None, :] * (ratio[:, None] - 1), 0)
    return W, s


def bodek_yang_part_weights(ev, knots, modes, grid, part):
    """x = 0: BY on in both the vector and axial parts, 1: BY off in the `part` half only (T2K's vector / axial split [G]); linear in x."""
    if grid not in _PDF: _PDF[grid] = GRV98(grid)
    s = np.isin(ev.am, modes) & (ev.Q2 > 0) & (ev.q0 > 0)
    on = dis_cross_section(_PDF[grid], ev, s, True, split=(True, True))
    off = np.maximum(dis_cross_section(_PDF[grid], ev, s, True, split=(part != "vector", part != "axial")), 0)
    good = on > 1e-12 * np.abs(on).max(); s[np.where(s)[0][~good]] = False; on, off = on[good], off[good]
    ratio = off / on
    W = np.ones((len(ev.mode), len(knots)))
    W[s] = np.maximum(1 + np.asarray(knots, float)[None, :] * (ratio[:, None] - 1), 0)
    return W, s


def r_long_weights(ev, knots, modes, grid, step=0.05):
    """sigma_L/sigma_T shifted by x * 0.05 (the BY uncertainty on R, arXiv:1011.6592), BY05 structure functions."""
    if grid not in _PDF: _PDF[grid] = GRV98(grid)
    s = np.isin(ev.am, modes) & (ev.Q2 > 0) & (ev.q0 > 0)
    nom = dis_cross_section(_PDF[grid], ev, s, True)
    good = nom > 1e-12 * np.abs(nom).max(); s[np.where(s)[0][~good]] = False; nom = nom[good]
    W = np.ones((len(ev.mode), len(knots)))
    for j, k in enumerate(knots):
        W[s, j] = np.maximum(dis_cross_section(_PDF[grid], ev, s, True, dR=k * step), 0) / nom
    return np.maximum(W, 0), s


# ---------------------------------------------------------------- the dial table
def compute(ev, item):
    """Knots and weights (n_events, n_knots) of one physics dial of the configuration, and the events it acts on."""
    r = item["response"]; knots = np.asarray(item["knots"], float)
    am, an, E = ev.am, ev.antinu, ev.E
    # MA_QE, QE_vector_FF: Llewellyn Smith nucleon current (Phys. Rep. 3 (1972) 261) contracted at the event's nucleon momenta; BBBA05 vector form factors (hep-ex/0602017); widths T2K (arXiv:2303.03222, arXiv:2606.14015)
    if r == "ma_qe": return knots, *ccqe_weights(ev, knots, "MA")
    if r == "vector_ff": return knots, *ccqe_weights(ev, knots, "VFF")
    # QE_Q2_0..7: CCQE normalisation in bins of true Q2, T2K (arXiv:2303.03222, arXiv:2606.14015)
    if r == "q2_norm":
        lo, hi = item["q2_range"]; return knots, *linear_class_weights((am == 1) & (ev.Q2 >= lo) & (ev.Q2 < hi), knots)
    # PB_C_nu, PB_C_nubar: global Fermi-surface cut on the pre-FSI nucleon momentum, T2K (arXiv:2606.14015) and GENIE's Fermi-gas k_F dial (arXiv:1510.05494)
    if r == "pauli_blocking":
        anti = item["antinu"]; s = (am == 1) & ev.carbon & (an == anti)
        p_out = ak.to_numpy(ak.fill_none(ak.max(ev.p_vert[ev.nuc_out], 1), 0.0))
        W = np.ones((len(am), len(knots))); W[s] = (p_out[s, None] >= knots[None, :]).astype(float)
        return knots, W, s
    # CC_norm_lowE: charged-current normalisation at 0.3-0.6 GeV, nu +2 % / nubar -1 % per unit, T2K (arXiv:2606.14015)
    if r == "cc_norm_lowE":
        s = (am < 30) & (E > 0.3) & (E < 0.6)
        W = np.ones((len(am), len(knots))); W[s] = 1 + np.where(an[s], -0.01, 0.02)[:, None] * knots[None, :]
        return knots, W, s
    # Nucleon_FSI: nucleon re-interaction strength with the inclusive count fixed, T2K (arXiv:2606.14015)
    if r == "nucleon_fsi": return knots, *nucleon_fsi_weights(ev, item["which"], knots)
    # QE_MINERvA_lowrecoil: MINERvA low-recoil Gaussian enhancement in (q0, q3) (arXiv:1803.09377)
    if r == "minerva_qe": return knots, *minerva_gauss_weights(ev, (am == 1) & ev.carbon, "qe", knots)
    # MEC_norm_nu, MEC_norm_nubar: 2p2h normalisation, T2K (arXiv:2303.03222, arXiv:2606.14015)
    if r == "mec_norm": return knots, *linear_class_weights((am == 2) & (an == item["antinu"]), knots)
    # MEC_pair_ratio_nu, MEC_pair_ratio_nubar: proton-neutron share of 2p2h (NEUT about 70 %, SuSAv2 about 80 %), T2K (arXiv:2606.14015)
    if r == "mec_pair_ratio":     # np share moved towards 80 % at x = 1 (NEUT ~70 %, SuSAv2 ~80 %: 2606.14015), 2p2h total fixed
        s = (am == 2) & ev.carbon; npair = pair_type(ev)
        f = np.zeros(len(am)); x = knots[None, :]; W = np.ones((len(am), len(knots)))
        for anti in ((False, True) if "antinu" not in item else (item["antinu"],)):
            m = s & (an == anti)
            if not m.any(): continue
            f0 = npair[m].mean(); a = 0.8 / f0 - 1
            W[m & npair] = 1 + x * a
            W[m & ~npair] = 1 - x * a * f0 / (1 - f0)
        if "antinu" in item: s = s & (an == item["antinu"])
        return knots, np.maximum(W, 0), s
    # MEC_MINERvA_np, MEC_MINERvA_nn: MINERvA low-recoil 2p2h tune restricted to one pair type (arXiv:1803.09377)
    if r == "minerva_2p2h":
        s = (am == 2) & ev.carbon & (pair_type(ev) == (item["pair"] == "np"))
        return knots, *minerva_gauss_weights(ev, s, "2p2h_" + item["pair"], knots)
    # MA_RES, CA5_RES: Delta(1232) axial form factor in Lalakulich & Paschos (PRD 71 074003, hep-ph/0501109), App. A and Eq. II.10; widths T2K (arXiv:2606.14015)
    if r == "ma_res": return knots, *res_axial_weights(ev, knots, "MA")
    if r == "ca5": return knots, *res_axial_weights(ev, knots, "CA5")
    # RES_Delta_decay: Delta decay angular distribution, Rein & Sehgal (Ann. Phys. 133 (1981) 79) Eq. 2.46 against isotropic, T2K (arXiv:2606.14015)
    if r == "delta_decay": return knots, *delta_decay_weights(ev, knots)
    # CC1pi0_norm_nu/nubar, CC_coh_norm, CC_misc_norm, CC_DIS_MPi_norm_nu/nubar: normalisation of NEUT modes, T2K (arXiv:2606.14015)
    if r == "mode_norm": return knots, *linear_class_weights(np.isin(ev.mode, item["modes"]), knots, item.get("nominal", 1.0))
    # RES_lowQ2_*: MINERvA low-Q2 suppression of resonant pion production, PRD 100 072005 (arXiv:1903.01558) Eqs. 6-7, Table IX
    if r == "lowq2_pion":
        R1, R2 = LOWQ2_CHANNELS[item["channel"]]
        return knots, *lowq2_pion_weights(ev, knots, R1, R2, lowq2_channel(ev, item["channel"]))
    # CC_coh_lowE: MINERvA coherent-pion tune (arXiv:1511.05944, arXiv:1803.09377)
    if r == "coh_lowE":   # MINERvA tune: coherent pions with T_pi < 0.45 GeV reduced by half at x = 1 (arXiv:1511.05944, 1803.09377)
        tpi = ak.to_numpy(ak.fill_none(ak.max(ev.T_fin[ev.pi_fin], 1), 99.0))
        s = (am == 16) & (tpi < 0.45)
        W = np.ones((len(am), len(knots))); W[s] = 1 - 0.5 * knots[None, :]
        return knots, W, s
    # MPi_multiplicity: multi-pion multiplicity, NEUT (x = 0) against AGKY (x = 1, arXiv:0904.4043), T2K (arXiv:2606.14015)
    if r == "multiplicity": return knots, *multiplicity_weights(ev, knots)
    # PiFSI_*: pion FSI fate fractions, T2K (arXiv:2303.03222, arXiv:2606.14015) with GENIE's fates (arXiv:1510.05494)
    if r == "pion_fsi": return knots, *pion_fsi_weights(ev, item["which"], knots)
    grid = os.path.expandvars(item.get("pdf_grid", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "auxiliary", "GRV98lo_patched.LHgrid")))
    # BY_DIS: Bodek-Yang correction (BY05) on at x = 0, off at x = 1, W > 2 GeV, T2K (arXiv:2303.03222); GRV98 LO parton densities
    if r == "bodek_yang": return knots, *bodek_yang_weights(ev, knots, item["modes"], grid)
    # DIS_R_long: R = sigma_L / sigma_T shifted by 0.05 per unit (arXiv:1011.6592)
    if r == "r_long": return knots, *r_long_weights(ev, knots, item["modes"], grid)
    # BY_MPi_vector, BY_MPi_axial: vector or axial part of the Bodek-Yang correction for multi-pion events, T2K (arXiv:2606.14015); axial K factors arXiv:2108.09240
    if r == "bodek_yang_part": return knots, *bodek_yang_part_weights(ev, knots, item["modes"], grid, item["part"])
    raise ValueError(f"unknown physics response {r}")
