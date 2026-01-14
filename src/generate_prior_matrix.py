import numpy as np
from scipy.ndimage import gaussian_filter
import ROOT
import argparse

def main():
    p = argparse.ArgumentParser(
        description="Generate a prior covariance matrix with localized correlations and save it to a ROOT file."
    )
    p.add_argument("--size", type=int, default=22, help="matrix size (NxN)")
    p.add_argument("--scale", type=float, default=1.0, help="Order of magnitude of sqrt(diag) elements")
    p.add_argument("--output_file", "-o", type=str, required=True, help="Output ROOT file name")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--filter", "-f", type=float, default=0.6, help="Sigma of Gaussian filter for smoothing")
    p.add_argument("--blobs", "-b", type=int, default=4, help="Number of localized correlated blobs to add")
    args = p.parse_args()

    scale = args.scale
    output_file_name = args.output_file
    seed = args.seed
    filter_sigma = args.filter
    blobs = args.blobs

    # matrix size
    N = args.size

    # set random seed for reproducibility
    np.random.seed(seed)
    # variance on diagonal
    diagonal = scale * np.random.normal(loc=1.0, scale=0.1, size=N)
    for i, d in enumerate(diagonal):
        while d <= 0:
            diagonal[i] = scale * np.random.normal(loc=1.0, scale=0.1, size=N)
    L = np.diag(diagonal)
    # add noise
    L += np.random.normal(loc=0.0, scale=scale*filter_sigma*0.17, size=(N, N))
    # add a few localized correlated blobs
    for _ in range(blobs):                    # number of blobs
        x = np.random.randint(0, N)
        y = np.random.randint(0, x)
        amp = np.random.uniform(-scale, scale)
        L[x, y] += amp
        if (amp < 0):
            print(f"Anticorrelation added at ({x}, {y}) with amplitude {amp:.3f}")
        else:
            print(f"Correlation added at ({x}, {y}) with amplitude {amp:.3f}")

    # smooth to create correlations (tune sigma for correlation length)
    L = gaussian_filter(L, sigma=filter_sigma)
    # add back some noise so it’s not too smooth
    L += np.random.normal(loc=0.0, scale=scale*filter_sigma*0.05, size=(N, N))
    # make it lower triangular
    L = np.tril(L)
    # compute covariance matrix. This is positive definite and symmetric
    M = L @ L.T

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