
from utils import *
import plot_utils
import argparse
from spline_generator import SplineGenerator
import ROOT
import yaml
from array import array


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Produce a GUNDAM input dataset from a NUISANCE flat tree.")
    p.add_argument("--config_spline", "-c", type=str, required=True, help="Path to the SPLINE configuration YAML file.")
    args = p.parse_args()

    # load splines from config file
    config_file = args.config_spline

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # top-level options
    generate_cov = cfg.get("generate_cov_matrix", False)

    # list of splines
    splines_dictionary = cfg.get("splines", [])

    if (not generate_cov):
        raise ValueError("generate_cov_matrix is set to False. This script only generates covariance matrices. Check your config file.")

    # save sigams for covariance matrix generation
    sigmas = []
    for item in splines_dictionary:
        name = item["name"]
        prior = item["prior"]
        sigma = item["sigma"]
        sigmas.append(sigma) # actually not variances, but sigmas
        importance = item["importance"]
        kind = item["type"]
        power = item["power"]
        npts = item["graph_points"]
        kin_var = item["kin_variable"]
        kmin = item["kin_min"]
        kmax = item["kin_max"]
        range_min = item.get("range_min", None)
        range_max = item.get("range_max", None)
        mirror_point = item.get("mirror_point", None)

        print(f"{name}")
        print(f"  prior        = {prior}")
        print(f"  sigma        = {sigma}")
        print(f"  importance   = {importance}")
        print(f"  dial range   = [{range_min}, {range_max}]")
        print(f"  graph_points = {npts}")
        print(f"  type         = {kind}")
        print(f"  power        = {power}")
        print(f"  kin_variable = {kin_var}")
        print(f"  kin var range= [{kmin}, {kmax}]")
        if (kind == "mirrored"):
            print(f"  mirror_point = {mirror_point}")
        # error if lower_bound, upper_bound, prior or sigma are missing and generate_cov_matrix is True
        if generate_cov:
            if "lower_bound" not in item or "upper_bound" not in item:
                raise ValueError(f"lower_bound and upper_bound must be specified for spline {name} when generate_cov_matrix is True.")
            if prior is None or sigma is None:
                raise ValueError(f"prior and sigma must be specified for spline {name} when generate_cov_matrix is True.")
        print()


    generate_cov = generate_cov
    # Generate covariance matrix if requested
    if generate_cov:
        print(f"Generating covariance matrix for the splined parameters...")
        filter_sigma = cfg.get("filter_sigma", 0.9)
        blobs = cfg.get("blobs", 6)
        seed = cfg.get("seed", 42)
        if len(sigmas) == 0:
            raise ValueError("No variances found for covariance matrix generation.")
        from generate_prior_matrix import build_posdef_matrix
        cov_matrix = build_posdef_matrix(np.array(sigmas), filter_sigma=filter_sigma, blobs=blobs, seed=seed)
        correl_matrix = ROOT.TMatrixT('double')(cov_matrix.GetNrows(), cov_matrix.GetNcols())
        for i in range(cov_matrix.GetNrows()):
            for j in range(cov_matrix.GetNcols()):
                correl_matrix[i, j] = cov_matrix[i, j] / (np.sqrt(cov_matrix[i, i]) * np.sqrt(cov_matrix[j, j])) if cov_matrix[i, i] > 0 and cov_matrix[j, j] > 0 else 0.0
        # save covariance matrix to ROOT file
        cov_out_file_name = path = os.path.expandvars(cfg["cov_file_path"])
        print(f"Saving covariance matrix to {cov_out_file_name}...")
        # open root file and save as objects: cov matrix, corr matrix, param names, priors and upper/lower bounds
        outfile_cov = ROOT.TFile(cov_out_file_name, "RECREATE")
        outfile_cov.WriteObject(cov_matrix, "prior_covariance_matrix")
        outfile_cov.WriteObject(correl_matrix, "prior_correlation_matrix")
        # param names
        param_names_list = [item["name"] for item in splines_dictionary]
        param_names = ROOT.TObjArray()  # container for TObjects
        for x in param_names_list:
            h = ROOT.TObjString(str(x))
            param_names.Add(h)
        outfile_cov.WriteObject(param_names, "param_names")
        # priors
        priors_list = [item["prior"] for item in splines_dictionary]
        priors = ROOT.TVectorD(len(priors_list), array('d', priors_list))
        outfile_cov.WriteObject(priors, "priors")
        # bounds
        lower_bounds_list = [item["lower_bound"] for item in splines_dictionary]
        upper_bounds_list = [item["upper_bound"] for item in splines_dictionary]
        lower_bounds = ROOT.TVectorD(len(lower_bounds_list), array('d', lower_bounds_list))
        upper_bounds = ROOT.TVectorD(len(upper_bounds_list), array('d', upper_bounds_list))
        outfile_cov.WriteObject(lower_bounds, "lower_bounds")
        outfile_cov.WriteObject(upper_bounds, "upper_bounds")

        print(f"Saved covariance matrix and metadata to {cov_out_file_name}")



