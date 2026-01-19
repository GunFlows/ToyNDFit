
// Opens a root file that contains a TMatrixD named "prior_covariance_matrix".
// Copy it over to an output file together with:
//  parameterNameTObjArray: param_names
//  parameterPriorTVectorD: priors
//  parameterLowerBoundsTVectorD: lower_bounds
//  parameterUpperBoundsTVectorD: upper_bounds

// Requires input matrix generated with src/generate_prior_matrix.py
// and saves other info reproducing the structure used in the gundam configuration

int write_param_names(){
  string infile_name = "spl6.root";
  string outfile_name = infile_name.substr(0,infile_name.find_last_of(".")) + "_with_names.root";


  TFile outfile(outfile_name.c_str(),"RECREATE");
  TFile* infile = new TFile(infile_name.c_str(),"READ");

  TMatrixD* cov_matrix = (TMatrixD*) infile->Get("prior_covariance_matrix");
  if (!cov_matrix) {
    cerr << "Error: Could not find 'prior_covariance_matrix' in " << infile_name << endl;
    return 1;
  }
  int cov_dimension = cov_matrix->GetNrows();
  cout << "Reading " << cov_dimension << "x" << cov_dimension << " cov matrix from " << infile_name << endl;


  vector<string> param_names = {
    "Spline_Enu",
    "Spline_PP",
    "Spline_FSI",
    "Spline_mirror_q0",
    "Spline_mirror_q3",
    "Spline_bound_W",
    // Add more parameter names as needed
  };

  vector<double> priors = {
    1.0, // Prior for Spline_Enu
    1.0, // Prior for Spline_PP
    1.0, // Prior for Spline_FSIstrength
    1.0, // Prior for Spline_mirror_q0
    1.0, // Prior for Spline_mirror_q3
    1.0, // Prior for Spline_bound_W
    // Add more priors as needed
  };

  vector<double> lower_bounds = {
    -999.0, // Lower bound for Spline_Enu
    -999.0, // Lower bound for Spline_PP
    -999.0, // Lower bound for Spline_FSIstrength
    -999.0, // Lower bound for Spline_mirror_q0
    -999.0, // Lower bound for Spline_mirror_q3
    -999.0, // Lower bound for Spline_bound_W
    // Add more lower bounds as needed
  };

  vector<double> upper_bounds = {
    999, // Upper bound for Spline_Enu
    999, // Upper bound for Spline_PP
    999, // Upper bound for Spline_FSIstrength
    999, // Upper bound for Spline_mirror_q0
    999, // Upper bound for Spline_mirror_q3
    999, // Upper bound for Spline_bound_W
    // Add more upper bounds as needed
  };


  int num_params = param_names.size();
  if (cov_dimension != num_params) {
    cerr << "Error: Dimension of covariance matrix (" << cov_dimension
         << ") does not match number of parameter names (" << num_params << ")." << endl;
    return 1;
  }


  // Convert vectors to ROOT objects and save to output file

  TObjArray* paramNameTObjArray = new TObjArray();
  for (const auto& name : param_names) {
    paramNameTObjArray->Add(new TObjString(name.c_str()));
  }
  TVectorD* parameterPriorTVectorD = new TVectorD(priors.size());
  for (size_t i = 0; i < priors.size(); ++i) {
    (*parameterPriorTVectorD)[i] = priors[i];
  }
  TVectorD* parameterLowerBoundsTVectorD = new TVectorD(lower_bounds.size());
  for (size_t i = 0; i < lower_bounds.size(); ++i) {
    (*parameterLowerBoundsTVectorD)[i] = lower_bounds[i];
  }
  TVectorD* parameterUpperBoundsTVectorD = new TVectorD(upper_bounds.size());
  for (size_t i = 0; i < upper_bounds.size(); ++i) {
    (*parameterUpperBoundsTVectorD)[i] = upper_bounds[i];
  }
  outfile.WriteObject(cov_matrix, "prior_covariance_matrix");
  outfile.WriteObject(paramNameTObjArray, "param_names");
  outfile.WriteObject(parameterPriorTVectorD, "priors");
  outfile.WriteObject(parameterLowerBoundsTVectorD, "lower_bounds");
  outfile.WriteObject(parameterUpperBoundsTVectorD, "upper_bounds");

  outfile.Close();
  infile->Close();

  cout << "Successfully wrote to " << outfile_name << endl;
  return 0;
}