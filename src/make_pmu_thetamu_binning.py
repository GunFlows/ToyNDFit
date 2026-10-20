import argparse
import numpy as np
import uproot


def main():
    p = argparse.ArgumentParser(
        description="Detector-systematics binning in (Pmu, CosThetamu), shared by all samples: a regular grid with edges at "
                    "equal-statistics quantiles of the selected events (finer version of PmuThetamuBins.txt)."
    )
    p.add_argument("--input_files", nargs="+", required=True, help="GUNDAM input trees (sample_sum) to take the selected events from")
    p.add_argument("--n_pmu", type=int, required=True, help="number of momentum bins")
    p.add_argument("--n_cos", type=int, required=True, help="number of cos(theta) bins")
    p.add_argument("--output_file", "-o", type=str, required=True, help="output binning file")
    args = p.parse_args()

    pmu, cos = [], []
    for f in args.input_files:
        a = uproot.open(f)["sample_sum"].arrays(["SelectedSample", "Pmu", "CosThetamu"], library="np")
        sel = a["SelectedSample"] > 0
        pmu.append(a["Pmu"][sel])
        cos.append(a["CosThetamu"][sel])
    pmu, cos = np.concatenate(pmu), np.concatenate(cos)

    # edges at quantiles, outer edges at the full range (Pmu in MeV/c)
    pe = np.unique(np.round(np.quantile(pmu, np.linspace(0, 1, args.n_pmu + 1)), 0))
    ce = np.unique(np.round(np.quantile(cos, np.linspace(0, 1, args.n_cos + 1)), 4))
    pe[0], pe[-1] = 0, 30000
    ce[0], ce[-1] = -1, 1

    with open(args.output_file, "w") as f:
        f.write("variables: Pmu Pmu CosThetamu CosThetamu\n")
        for i in range(len(pe) - 1):
            for j in range(len(ce) - 1):
                f.write(f"{pe[i]:g} {pe[i + 1]:g} {ce[j]:g} {ce[j + 1]:g}\n")

    counts = np.histogram2d(pmu, cos, [pe, ce])[0]
    print(f"{(len(pe) - 1) * (len(ce) - 1)} bins ({len(pe) - 1} x {len(ce) - 1}) written to {args.output_file}; "
          f"selected events per bin: min {counts.min():.0f}, median {np.median(counts):.0f}, empty {(counts == 0).sum()}")


if __name__ == "__main__":
    main()
