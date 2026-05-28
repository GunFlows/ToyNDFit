#!/usr/bin/env python3
"""2D demo: marginal *PEAK* shifts away from the MAP, without truncation.

Goal:
    Replicate the T2K-100p10 detector pattern, where every DET marginal
    *peak* sits ~+0.05 to the right of the MAP, while the 1D PROFILE
    (slice through the MAP) peaks AT the MAP.

What previous attempts got wrong:
    (i) Plain Poisson with linear bins: gives a tail-induced *mean* shift
        but barely moves the *peak*.
    (ii) Hard truncation: shifts the peak but the cliff is unphysical
        for 100p10 and just chops a tail off.

What this toy uses (smooth LH, no truncation):
    A Poisson model with a *non-linear cross-bin* prediction. The LH then
    has a curved valley (ridge) in (x,y) — not just a quadratic well.
    Combined with a correlated prior whose aligned eigenmode runs along
    the same direction as the curved ridge, the joint posterior is
    skewed. The per-x slice volume depends on x, and the marginal peak
    no longer coincides with the MAP.

Model:
    Two parameters x, y > 0.
    Three Poisson bins:
        b1: lambda_1 = N1 * x                 -> constrains x
        b2: lambda_2 = N2 * y                 -> constrains y
        b3: lambda_3 = N3 * (x*y)**0.5        -> cross, NON-LINEAR
    Low N_i exaggerates the Poisson asymmetry.

    Prior: bivariate Gaussian, mean (1, 1), strong positive correlation.

Outputs (./out/):
    posterior_2d_with_marginals.png
    report.txt
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# --- model ------------------------------------------------------------------
# Low-stat Poisson bins; data sits at nominal (x=y=1) so LH gradient = 0 there.
N1, N2, N3 = 0.7, 0.7, 1.5
d1, d2, d3 = 1, 1, 2   # rounded counts at x=y=1 (nominal predictions are 0.7, 0.7, 1.5)

# Strong positive prior correlation (aligned eigenvector ~ (1,1)/sqrt(2))
mu_x, mu_y = 1.0, 1.0
sigma = 2.0
rho = 0.95
cov_prior = sigma**2 * np.array([[1.0, rho], [rho, 1.0]])
inv_cov = np.linalg.inv(cov_prior)


def poisson_nll(lam: np.ndarray, d: float) -> np.ndarray:
    """-log Poisson(d | lam), dropping the d! constant.
    Smooth everywhere lam>0; no truncation."""
    out = np.full_like(lam, np.inf, dtype=float)
    pos = lam > 0
    l = lam[pos]
    out[pos] = l - d * np.log(l)
    return out


def neg_log_post(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    XY = np.clip(X * Y, 1e-300, None)
    l1 = N1 * X
    l2 = N2 * Y
    l3 = N3 * np.sqrt(XY)
    nll = poisson_nll(l1, d1) + poisson_nll(l2, d2) + poisson_nll(l3, d3)
    dx, dy = X - mu_x, Y - mu_y
    nlp = 0.5 * (dx * dx * inv_cov[0, 0]
                 + 2.0 * dx * dy * inv_cov[0, 1]
                 + dy * dy * inv_cov[1, 1])
    return nll + nlp


# --- grid -------------------------------------------------------------------
NG = 1601
x_grid = np.linspace(0.05, 3.0, NG)
y_grid = np.linspace(0.05, 3.0, NG)
dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]
X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")

NLP = neg_log_post(X, Y)
NLP = NLP - np.nanmin(NLP[np.isfinite(NLP)])
post = np.where(np.isfinite(NLP), np.exp(-NLP), 0.0)
Z = post.sum() * dx * dy
post /= Z

# --- MAP --------------------------------------------------------------------
iy_map, ix_map = np.unravel_index(np.argmax(post), post.shape)
x_map, y_map = x_grid[ix_map], y_grid[iy_map]

# --- marginals --------------------------------------------------------------
px = post.sum(axis=0) * dy
py = post.sum(axis=1) * dx

x_marg_peak = x_grid[np.argmax(px)]
y_marg_peak = y_grid[np.argmax(py)]
x_marg_mean = float(np.sum(x_grid * px) * dx)
y_marg_mean = float(np.sum(y_grid * py) * dy)

# --- profile (slice through MAP) -------------------------------------------
prof_x = post[iy_map, :]
prof_y = post[:, ix_map]
prof_x_peak = x_grid[np.argmax(prof_x)]
prof_y_peak = y_grid[np.argmax(prof_y)]
prof_x_norm = prof_x / (prof_x.sum() * dx)
prof_y_norm = prof_y / (prof_y.sum() * dy)

# --- prior eigendecomposition ----------------------------------------------
ev_prior, evec_prior = np.linalg.eigh(cov_prior)
v_aligned = evec_prior[:, 1]
if v_aligned.sum() < 0:
    v_aligned = -v_aligned


# --- report -----------------------------------------------------------------
report_lines = [
    "=== 2D PEAK-shift toy: smooth Poisson + correlated prior, no truncation ===",
    "",
    "LH:  3 Poisson bins   lam = (N1*x, N2*y, N3*sqrt(x*y))",
    f"     N = ({N1}, {N2}, {N3})    d = ({d1}, {d2}, {d3})",
    f"Prior:  Bivariate Gaussian, mu=({mu_x}, {mu_y}), sigma={sigma}, rho={rho}",
    f"        eigenvalues: {ev_prior[::-1]}",
    f"        aligned eigenvector: {v_aligned}",
    "",
    f"MAP                       : ({x_map:.4f}, {y_map:.4f})",
    f"Profile-peak of x (slice y=y_MAP) : {prof_x_peak:.4f}  shift = {prof_x_peak - x_map:+.4f}",
    f"Profile-peak of y (slice x=x_MAP) : {prof_y_peak:.4f}  shift = {prof_y_peak - y_map:+.4f}",
    f"Marginal-PEAK of x        : {x_marg_peak:.4f}  shift = {x_marg_peak - x_map:+.4f}    <- KEY",
    f"Marginal-PEAK of y        : {y_marg_peak:.4f}  shift = {y_marg_peak - y_map:+.4f}    <- KEY",
    f"Marginal-mean of x        : {x_marg_mean:.4f}  shift = {x_marg_mean - x_map:+.4f}",
    f"Marginal-mean of y        : {y_marg_mean:.4f}  shift = {y_marg_mean - y_map:+.4f}",
]
(OUT / "report.txt").write_text("\n".join(report_lines))
print("\n".join(report_lines))


# --- plot -------------------------------------------------------------------
fig = plt.figure(figsize=(10, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.5], height_ratios=[1.5, 4],
                      hspace=0.08, wspace=0.08, left=0.09, right=0.97,
                      top=0.93, bottom=0.08)

ax2d = fig.add_subplot(gs[1, 0])
ax2d.contourf(X, Y, post, levels=22, cmap="Blues")
levels = np.percentile(post[post > 0], [50, 70, 85, 93, 97, 99, 99.7])
levels = sorted(set(float(l) for l in levels if l > 0))
ax2d.contour(X, Y, post, levels=levels, colors="k", linewidths=0.6, alpha=0.5)

ax2d.plot([x_map], [y_map], "o", color="red", ms=10, label="MAP (joint mode)")
ax2d.plot([x_marg_peak], [y_marg_peak], "s", color="orange", ms=10,
          label="(marg_x peak, marg_y peak)")

arrow_scale = 1.4 * np.sqrt(ev_prior[1])
ax2d.annotate("", xy=(mu_x + arrow_scale * v_aligned[0], mu_y + arrow_scale * v_aligned[1]),
              xytext=(mu_x, mu_y),
              arrowprops=dict(arrowstyle="->", color="purple", lw=1.6))
ax2d.text(mu_x + 1.05 * arrow_scale * v_aligned[0],
          mu_y + 1.05 * arrow_scale * v_aligned[1],
          " aligned\n eigenmode", color="purple", fontsize=9)
ax2d.set_xlabel("x"); ax2d.set_ylabel("y")
ax2d.set_xlim(x_grid[0], x_grid[-1]); ax2d.set_ylim(y_grid[0], y_grid[-1])
ax2d.legend(loc="upper left", fontsize=8)
ax2d.grid(True, alpha=0.2)

axx = fig.add_subplot(gs[0, 0], sharex=ax2d)
axx.plot(x_grid, px, color="C0", lw=2.0, label="marginal p(x)")
axx.plot(x_grid, prof_x_norm, color="C2", lw=1.2, ls="--",
         label="profile p(x|y=y_MAP)")
axx.axvline(x_map, color="red", lw=1.5,
            label=f"MAP x={x_map:.3f}")
axx.axvline(x_marg_peak, color="orange", lw=1.5,
            label=f"marg peak x={x_marg_peak:.3f}  Δ={x_marg_peak-x_map:+.3f}")
axx.set_ylabel("p(x)")
axx.tick_params(axis="x", labelbottom=False)
axx.legend(loc="upper right", fontsize=7)
axx.grid(True, alpha=0.2)

axy = fig.add_subplot(gs[1, 1], sharey=ax2d)
axy.plot(py, y_grid, color="C0", lw=2.0, label="marginal p(y)")
axy.plot(prof_y_norm, y_grid, color="C2", lw=1.2, ls="--",
         label="profile p(y|x=x_MAP)")
axy.axhline(y_map, color="red", lw=1.5)
axy.axhline(y_marg_peak, color="orange", lw=1.5)
axy.set_xlabel("p(y)")
axy.tick_params(axis="y", labelleft=False)
axy.legend(loc="upper right", fontsize=7)
axy.grid(True, alpha=0.2)

title = (
    f"2D toy: smooth Poisson LH + correlated prior (no truncation)\n"
    f"N=({N1},{N2},{N3}), d=({d1},{d2},{d3}),  prior σ={sigma}, ρ={rho};   "
    f"peak shift ≈ ({x_marg_peak-x_map:+.3f}, {y_marg_peak-y_map:+.3f})"
)
fig.suptitle(title, fontsize=11, y=0.97)
out = OUT / "posterior_2d_with_marginals.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"\nwrote {out}")
