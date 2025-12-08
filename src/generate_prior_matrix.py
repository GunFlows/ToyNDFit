import numpy as np
from scipy.ndimage import gaussian_filter
import ROOT
import argparse

def main():
    p = argparse.ArgumentParser(
        description="Generate a prior covariance matrix with localized correlations and save it to a ROOT file."
    )
    p.add_argument("--scale", type=float, default=1.0, help="Covariance scale factor")
    p.add_argument("--output_file", "-o", type=str, required=True, help="Output ROOT file name")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()

    scale = args.scale
    output_file_name = args.output_file
    seed = args.seed

    # matrix size
    N = 22

    # set random seed for reproducibility
    np.random.seed(seed)
    # variance on diagonal
    diagonal = scale * np.random.normal(loc=1.0, scale=0.1, size=N)
    M = np.diag(diagonal)
    # add noise
    M += np.random.normal(loc=0.0, scale=scale*0.1, size=(N, N))
    # add a few localized correlated blobs
    for _ in range(4):                    # number of blobs
        x = np.random.randint(0, N)
        y = np.random.randint(x, N)
        amp = np.random.uniform(-scale, scale)
        M[x, y] += amp
        # print(f"Added blob at ({x}, {y}) with amplitude {amp:.3f}")
    # smooth to create correlations (tune sigma for correlation length)
    M = gaussian_filter(M, sigma=0.6)
    # add back some noise so it’s not too smooth
    M += np.random.normal(loc=0.0, scale=scale*0.03, size=(N, N))
    # symmetrize
    for i in range(N):
        for j in range(i + 1, N):
            M[j, i] = M[i, j]

    # convert to TMatrixT<double> format and save to root file
    matrix = ROOT.TMatrixT('double')(N, N)
    for i in range(N):
        for j in range(N):
            matrix[i][j] = M[i, j]

    # save matrix to png for visualization
    ROOT.gROOT.SetBatch(True)  # run in batch mode to avoid GUI
    c = ROOT.TCanvas('c', 'Prior Covariance Matrix', 800, 600)
    c.cd()
    c.SetRightMargin(0.15)
    matrix.Draw("colz")  # optional: visualize the matrix
    # remove stats box
    ROOT.gStyle.SetOptStat(0)
    c.Update()
    c.SaveAs(output_file_name.replace(".root", ".png"))
    # save to root file
    output_file = ROOT.TFile(output_file_name, 'RECREATE')
    matrix.Write('prior_covariance_matrix')
    output_file.Close()

    print(f"Covariance matrix saved to {output_file_name}")


if __name__ == "__main__":
    main()