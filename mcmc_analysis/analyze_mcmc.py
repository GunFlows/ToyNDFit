#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re
import ROOT
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def list_dir_keys(d):
    keys = d.GetListOfKeys()
    out = []
    if not keys:
        return out
    for i in range(keys.GetSize()):
        k = keys.At(i)
        out.append((str(k.GetName()), int(k.GetCycle()), str(k.GetClassName())))
    return out

def walk_dirs(tfile):
    dirs = []
    def rec(d, path):
        keys = d.GetListOfKeys()
        if not keys:
            return
        for i in range(keys.GetSize()):
            k = keys.At(i)
            if str(k.GetClassName()).startswith("TDirectory"):
                sub = d.Get(k.GetName())
                if sub:
                    subpath = f"{path}/{k.GetName()}" if path else str(k.GetName())
                    dirs.append(subpath)
                    rec(sub, subpath)
    rec(tfile, "")
    return dirs

def find_dirs_containing(tfile, objname):
    out = []
    for dpath in walk_dirs(tfile):
        d = tfile.GetDirectory(dpath)
        if not d:
            continue
        for name, cyc, cname in list_dir_keys(d):
            if name == objname:
                out.append(dpath)
                break
    return out

def infer_base_dir(tfile):
    pdirs = set(find_dirs_containing(tfile, "parameterSets"))
    mdirs = set(find_dirs_containing(tfile, "MCMC"))
    inter = sorted(list(pdirs & mdirs), key=lambda s: s.count("/"), reverse=True)
    return inter[0]

def get_latest_obj_in_dir(tfile, dirname, objname):
    d = tfile.GetDirectory(dirname)
    best = -1
    for name, cyc, _ in list_dir_keys(d):
        if name == objname:
            best = max(best, cyc)
    obj = d.Get(f"{objname};{best}")
    return obj, best

def read_param_names(tfile, base_dir):
    t, cyc = get_latest_obj_in_dir(tfile, base_dir, "parameterSets")
    t.GetEntry(0)
    v = getattr(t, "parameterName")  # vector<string>
    names = [str(v.at(i)) for i in range(v.size())]
    return names, cyc

def read_points(t_mcmc, max_steps=None):
    n = int(t_mcmc.GetEntries())
    if max_steps is not None:
        n = min(n, int(max_steps))
    if not t_mcmc.GetBranch("Points"):
        raise RuntimeError("MCMC tree has no 'Points' branch.")
    t_mcmc.GetEntry(0)
    d = int(getattr(t_mcmc, "Points").size())
    pts = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        t_mcmc.GetEntry(i)
        v = getattr(t_mcmc, "Points")
        for j in range(d):
            pts[i, j] = float(v.at(j))
    return pts

def parse_dims(spec, ndim):
    """
    spec examples:
      "0,1,2"
      "0:20"  (0..19)
      "0:20,33,40:45"
    """
    if not spec:
        return list(range(ndim))
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            a, b = chunk.split(":")
            a = int(a) if a else 0
            b = int(b) if b else ndim
            out.extend(list(range(max(0, a), min(ndim, b))))
        else:
            out.append(int(chunk))
    seen = set()
    out2 = []
    for i in out:
        if 0 <= i < ndim and i not in seen:
            seen.add(i)
            out2.append(i)
    return out2

def trace_plot(x, outpath, burn):
    plt.figure()
    plt.plot(x, linewidth=0.7)
    if burn > 0:
        plt.axvline(burn, linestyle="--")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def heatmap(mat, labels, outpath, title):
    d = len(labels)
    plt.figure(figsize=(max(6, 0.35*d), max(5, 0.35*d)))
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xticks(range(d), labels, rotation=90)
    plt.yticks(range(d), labels)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def corner(samples, labels, outpath, bins=40):
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.2*d, 2.2*d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], bins=bins)
            elif i > j:
                ax.hist2d(samples[:, j], samples[:, i], bins=bins)
            else:
                ax.axis("off")
            if i == d-1 and i >= j:
                ax.set_xlabel(labels[j], rotation=90)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--burnin-frac", type=float, default=0.1)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--dims", default="", help="Dim list/ranges: '0:20,33,40:45' (default: all)")
    ap.add_argument("--select", default="", help="Regex on parameter names (optional)")
    ap.add_argument("--corner-max", type=int, default=25, help="Max dims for corner plot (default 20)")
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()
    ensure_dir(args.outdir)

    tf = ROOT.TFile.Open(args.input, "READ")

    base_dir = infer_base_dir(tf)
    names, pcyc = read_param_names(tf, base_dir)
    t_mcmc, mcyc = get_latest_obj_in_dir(tf, base_dir, "MCMC")

    pts = read_points(t_mcmc, args.max_steps)
    nsteps, ndim = pts.shape

    burn = int(max(0, min(nsteps, args.burnin_frac * nsteps)))
    thin = max(1, int(args.thin))

    sel = parse_dims(args.dims, min(ndim, len(names)))

    if args.select.strip():
        rx = re.compile(args.select)
        sel = [i for i in sel if rx.search(names[i])]
        if not sel:
            raise RuntimeError(f"--select matched no parameters: {args.select}")

    labels = [names[i] for i in sel]
    labels_short = [s if len(s) <= 60 else s[:57] + "..." for s in labels]

    post = pts[burn:nsteps:thin][:, sel]
    if post.shape[0] < 2:
        raise RuntimeError("Not enough post-burn-in samples. Reduce burnin/thin or run more steps.")

    with open(os.path.join(args.outdir, "report.txt"), "w") as f:
        f.write(f"input: {args.input}\n")
        f.write(f"base_dir: {base_dir}\n")
        f.write(f"parameterSets cycle: {pcyc}\n")
        f.write(f"MCMC cycle: {mcyc}\n")
        f.write(f"steps read: {nsteps}\n")
        f.write(f"burn: {burn}\n")
        f.write(f"thin: {thin}\n")
        f.write(f"selected dims: {len(sel)}\n")

    cov = np.cov(post, rowvar=False, ddof=1)
    corr = np.corrcoef(post, rowvar=False)
    np.save(os.path.join(args.outdir, "cov.npy"), cov)
    np.save(os.path.join(args.outdir, "corr.npy"), corr)

    show_heat = min(len(sel), 50)
    heatmap(cov[:show_heat, :show_heat],
            labels_short[:show_heat],
            os.path.join(args.outdir, "covariance.png"),
            f"Posterior covariance (first {show_heat} dims)")

    heatmap(corr[:show_heat, :show_heat],
            labels_short[:show_heat],
            os.path.join(args.outdir, "correlation.png"),
            f"Posterior correlation (first {show_heat} dims)")

    for k, idx in enumerate(sel[:6]):
        trace_plot(pts[:, idx], os.path.join(args.outdir, f"trace_param_{k}_{idx}.png"), burn)

    cd = min(len(sel), int(args.corner_max))
    corner(post[:, :cd], labels_short[:cd], os.path.join(args.outdir, "corner.png"), bins=args.bins)


if __name__ == "__main__":
    main()
