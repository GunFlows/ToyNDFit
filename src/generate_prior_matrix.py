import numpy as np
from scipy.ndimage import gaussian_filter
import ROOT
import argparse


def build_posdef_matrix(diagonal, filter_sigma=.9, scale = 0.5, blobs=6, seed=42, no_noise=False, symmetric_blobs=False) -> ROOT.TMatrixT:
    """
    Build a positive definite covariance matrix with localized correlations.

    Parameters
    ----------
    diagonal : np.ndarray
        1D array representing the diagonal elements (variances) of the covariance matrix.
    filter_sigma : float
        Sigma of Gaussian filter for smoothing.
    scale : float
        Scale of the correlations and the noise (if applicable)
    blobs : int
        Number of localized correlated blobs to add.
    seed : int
        Random seed for reproducibility.
    no_noise : bool
        If True, do not add random noise to the matrix.
    symmetric_blobs : bool
        If True, force alternating sign on blob amplitudes (sign flips every blob)
        and bias magnitudes toward [scale/2, scale]. This breaks the bias toward
        a strongly aligned dominant eigenvector, so the resulting covariance has
        eigenmodes with mixed-sign components (low alignment), suppressing the
        correlation-driven marginal-peak-shift effect.

    Returns
    -------
    np.ndarray
        Positive definite covariance matrix.
    """
    N = len(diagonal)
    np.random.seed(seed)

    L = np.diag(diagonal)


    print(f"DEBUG: Building matrix of size {N}x{N} with diag elements: {diagonal}")

    # add noise
    if (not no_noise):
        L += np.random.normal(loc=0.0, scale=scale*0.03, size=(N, N))
    # add a few localized correlated blobs
    for i in range(blobs):                    # number of blobs
        x = np.random.randint(0, N)
        if (x==0):
            y = 0
        else:
            y = np.random.randint(0, x)
        if symmetric_blobs:
            # Alternate sign blob-by-blob; magnitude in [scale/2, scale]
            sign = 1.0 if (i % 2 == 0) else -1.0
            amp = sign * np.random.uniform(0.5 * scale, scale)
        else:
            amp = np.random.uniform(-scale, scale)
        L[x, y] += amp
        if (amp < 0):
            print(f"Anticorrelation added at ({x}, {y}) with amplitude {amp:.3f}")
        else:
            print(f"Correlation added at ({x}, {y}) with amplitude {amp:.3f}")

    # smooth to create correlations (tune sigma for correlation length)
    L = gaussian_filter(L, sigma=filter_sigma)
    # add back some noise so it’s not too smooth
    if (not no_noise):
        L += np.random.normal(loc=0.0, scale=scale*0.01, size=(N, N))
    # make it lower triangular
    L = np.tril(L)
    # compute covariance matrix. This is positive definite and symmetric by definition
    M = L @ L.T
    # to avoid the nose to sum up
    d = np.sqrt(np.diag(M)); M = M * np.outer(diagonal / d, diagonal / d)

    # check that M is positive definite
    eigvals = np.linalg.eigvalsh(M)
    print(f"Minimum eigenvalue of covariance matrix: {np.min(eigvals)}")
    if np.any(eigvals <= 0):
        raise ValueError("Covariance matrix is not positive definite!")

    # convert to TMatrixT<double> format and save to root file
    matrix = ROOT.TMatrixT('double')(N, N)
    for i in range(N):
        for j in range(N):
            matrix[i][j] = M[i, j]

    return matrix # return the ROOT TMatrixT



def main():
    p = argparse.ArgumentParser(
        description="Generate a prior covariance matrix with localized correlations and save it to a ROOT file."
    )
    p.add_argument("--size", type=int, default=22, help="matrix size (NxN)")
    p.add_argument("--scale", type=float, default=1.0, help="Order of magnitude of sqrt(diag) elements")
    p.add_argument("--diagonal", type=str, help="You can also provide a file with diagonal sqrt(elements), one per line. This overrides size and scale.")
    p.add_argument("--output_file", "-o", type=str, required=True, help="Output ROOT file name")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--filter", "-f", type=float, default=0.6, help="Sigma of Gaussian filter for smoothing")
    p.add_argument("--blobs", "-b", type=int, default=4, help="Number of localized correlated blobs to add")
    p.add_argument("--no-noise", default=False, help="Do not add noise", action='store_true')
    p.add_argument("--symmetric-blobs", default=False, action='store_true',
                   help="Force alternating sign on blob amplitudes so the resulting covariance has eigenvectors with mixed-sign components (low alignment).")

    args = p.parse_args()

    scale = args.scale
    output_file_name = args.output_file
    seed = args.seed
    filter_sigma = args.filter
    blobs = args.blobs

    no_noise = args.no_noise
    symmetric_blobs = args.symmetric_blobs

    # matrix size
    N = args.size


    # set random seed for reproducibility
    np.random.seed(seed)

    if args.diagonal:
        diagonal = np.loadtxt(args.diagonal)
        N = len(diagonal)
        print(f"Using provided diagonal with size {N}")
    else:
        # variance on diagonal
        if no_noise:
            diagonal = np.full(N, scale)
        else:
            diagonal = scale * np.random.normal(loc=1.0, scale=0.1*scale, size=N)
            # avoid negatives (although unlikely)
            for i, d in enumerate(diagonal):
                while d <= 0:
                    diagonal[i] = scale * np.random.normal(loc=1.0, scale=0.1*scale)


    # build positive definite covariance matrix
    matrix = build_posdef_matrix(diagonal, filter_sigma=filter_sigma, scale=scale, blobs=blobs, seed=seed, no_noise=no_noise, symmetric_blobs=symmetric_blobs)

    # compute correlation matrix for visualization
    corr_matrix = ROOT.TMatrixT('double')(N, N)
    for i in range(N):
        for j in range(N):
            corr_matrix[i][j] = matrix[i, j] / np.sqrt(matrix[i, i] * matrix[j, j])

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
    corr_matrix.Write('prior_correlation_matrix')
    output_file.Close()

    print(f"Covariance matrix saved to {output_file_name}")


if __name__ == "__main__":
    main()