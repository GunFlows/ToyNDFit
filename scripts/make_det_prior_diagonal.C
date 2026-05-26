// Build a 40x40 detector covariance matrix that is DIAGONAL (zero correlations)
// with the same per-parameter sigma profile as prior_detector.root.
//
// Output: GundamWorkspace/inputs/parameters/linear/det_prior_diagonal.root
//
// Usage:
//   root -l -b -q ToyNDFit/scripts/make_det_prior_diagonal.C
//
// Run from the ToyNDFit root (uses relative paths).

void make_det_prior_diagonal() {
  const char* in_path  = "GundamWorkspace/inputs/parameters/linear/prior_detector.root";
  const char* out_path = "GundamWorkspace/inputs/parameters/linear/det_prior_diagonal.root";

  TFile *fin = TFile::Open(in_path);
  if (!fin || fin->IsZombie()) {
    printf("ERROR: cannot open %s\n", in_path);
    return;
  }
  TMatrixD *cov_in = (TMatrixD*) fin->Get("prior_covariance_matrix");
  if (!cov_in) {
    printf("ERROR: no prior_covariance_matrix in %s\n", in_path);
    return;
  }
  int n = cov_in->GetNrows();

  TMatrixD cov_out(n, n);
  TMatrixD corr_out(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      cov_out(i, j)  = (i == j) ? (*cov_in)(i, i) : 0.0;
      corr_out(i, j) = (i == j) ? 1.0 : 0.0;
    }
  }

  printf("Diagonal covariance built (%dx%d). sigma min/mean/max = %.4f / %.4f / %.4f\n",
         n, sqrt(cov_out(0,0)), 0.0, 0.0);  // placeholder; real stats below
  double dmin=1e99, dmax=-1e99, dsum=0;
  for (int i=0;i<n;++i){ double d = cov_out(i,i); dsum+=d; if(d<dmin)dmin=d; if(d>dmax)dmax=d; }
  printf("  -> sigma range [%.4f, %.4f] mean %.4f\n", sqrt(dmin), sqrt(dmax), sqrt(dsum/n));
  printf("  -> all off-diagonals = 0\n");

  TFile *fout = new TFile(out_path, "RECREATE");
  cov_out.Write("prior_covariance_matrix");
  corr_out.Write("prior_correlation_matrix");
  fout->Close();

  fin->Close();
  printf("Wrote %s\n", out_path);
}
