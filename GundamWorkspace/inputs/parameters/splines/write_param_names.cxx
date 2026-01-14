
// open a root file that contains a TMatrixD named "prior_covariance_matrix". Copy it over to an output file together with:
//  parameterNameTObjArray: param_names
//  parameterPriorTVectorD: priors
//  parameterLowerBoundsTVectorD: lower_bounds
//  parameterUpperBoundsTVectorD: upper_bounds

int write_param_names(){

  string outfile_name = "spline_covariance_matrix_with_names.root";
  TFile outfile(outfile_name.c_str(),"RECREATE");
  TFile* infile = new TFile("matrix.root","READ");
  TMatrixD* cov_matrix = (TMatrixD*) infile->Get("prior_covariance_matrix");


  vector<string> param_names = {
    "Spline_Enu",
    "Spline_PP",
    "Spline_FSIstrength",
    // Add more parameter names as needed
  };

  vector<double> priors = {
    1.0, // Prior for Spline_Enu
    1.0, // Prior for Spline_PP
    1.0, // Prior for Spline_FSIstrength
    // Add more priors as needed
  };

  vector<double> lower_bounds = {
    -999.0, // Lower bound for Spline_Enu
    -999.0, // Lower bound for Spline_PP
    -999.0, // Lower bound for Spline_FSIstrength
    // Add more lower bounds as needed
  };

  vector<double> upper_bounds = {
    999, // Upper bound for Spline_Enu
    999, // Upper bound for Spline_PP
    999, // Upper bound for Spline_FSIstrength
    // Add more upper bounds as needed
  };





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

  cout << "Success. Wrote to " << outfile_name << endl;
  return 0;
}