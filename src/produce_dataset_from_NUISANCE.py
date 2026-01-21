
from utils import *
import plot_utils
import argparse
from spline_generator import SplineGenerator
import ROOT
import yaml
from array import array

# Function to read the NUISANCE flat trees, smear the momenta based on a realistic momentum and angle resolution, and
# generates a "GUNDAM dataset": basically a root file with a TTree containing FS kinematics.
# One branch of the tree is "SelectedSample": a simple signal selection should be performed.



# Minimal version: produce only branches "SelectedSample", "Pmu", "CosThetamu".



def make_dataset_from_nuisance(file_path, isFHC = True, max_events = -1, tree_name="FlatTree_VARS", splines_dict = None):
    """
    Read a NUISANCE flat tree and return a list of NUISANCEEvent objects.
    This is just a test function!

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the NUISANCE flat tree.
    tree_name : str
        Name of the tree to read (default is "flat_tree").
    isFHC : bool
        True if FHC beam mode, False if RHC.
    max_events : int
        Maximum number of events to process. Default is -1 (all events).
    splines_dict : dict
        Dictionary containing the spline configurations.

    Returns
    -------
    dictionary : dict
        A dictionary of lists, each list containing the values of a certain computed variable.
        To add a variable, just create a new list, fill it in the loop over events, and add it to the dictionary at the end.
    """

    # open pmu and costhetamu resolution maps
    reso_maps_folder = os.environ.get("RESO_MAPS_FOLDER")
    if not reso_maps_folder:
        raise EnvironmentError("Can't find resolution maps because environment variable `RESO_MAPS_FOLDER` is not set. Did you forget to run setup.sh?")

    pmu_map   = uproot.open(os.path.join(reso_maps_folder, "pmu_resolution_map.root"))["pmu_resolution_map"]
    cthetamap = uproot.open(os.path.join(reso_maps_folder, "costhetamu_resolution_map.root"))["costhetamu_resolution_map"]

    start_time = datetime.datetime.now()
    f = NUISANCEFile(file_path, tree_name, relevant_keys=None, step=10000)
    num_entries = len(f)
    if (max_events < 0):
        max_events = num_entries

    # OUTPUT BRANCHES LISTS
    SelectedSample_list = []
    Pmu_list = []
    CosThetamu_list = []
    TruePmu_list = []
    TrueCosThetamu_list = []
    Reaction_list = []
    Enu_list = []
    # initialize an empty list for each of the splines in the configuration dictionary
    Splines_lists = {}
    for item in splines_dict:
        name = item["name"]
        Splines_lists[name] = []

    # Generate splines according to the configuration dictionary
    Spline_objects = {}
    for item in splines_dict:
        name = item["name"]
        kind = item["type"]
        power = item["power"]
        prior = item["prior"]
        kin_var = item["kin_variable"]
        kmin = item["kin_min"]
        kmax = item["kin_max"]
        imp = item["importance"]
        mirror_point = item.get("mirror_point", None)

        Spline_objects[name] = SplineGenerator(spline_type=kind, prior=prior, power=power, range_width=imp, kin_min=kmin, kin_max=kmax, mirror_point=mirror_point)


    pbar = tqdm(total=max_events)
    for i, event in enumerate(f):
        # limit number of events for testing
        if i > max_events and max_events > 0:
            break
        # progress bar
        if (i + 1) % (max_events // 50) == 0:
            pbar.update(max_events // 50)
        # Perform selection
        sample = event.PerformSelection(isFHC)

        # compute kinematics and variables
        Pmu = event.Pmu()
        CosThetamu = event.CosThetamu()
        Enu = event.Enu_true
        # smear Pmu, CosThetamu
        Pmu_smeared, CosThetamu_smeared = smear(Pmu, CosThetamu, pmu_map, cthetamap)

        # is pion production event?
        isPP = event.isPP()
        pion_energy = event.total_pion_energy() if isPP else 0.0

        # generate splines graphs
        for item in splines_dict:
            name = item["name"]
            kin_var = item["kin_variable"]
            npts = item.get("graph_points", 10)
            kmin = item.get("kin_min", 0.5)
            kmax = item.get("kin_max", 1.5)
            range_min = item.get("range_min", 0.5)
            range_max = item.get("range_max", 1.5)
            prior = item["prior"] # just to check that the prior has w=1 and it's in the dial range

            Spline_obj = Spline_objects[name]

            kin_val = getattr(event, kin_var)() if callable(getattr(event, kin_var, None)) else getattr(event, kin_var, None)
            if kin_val is None:
                raise ValueError(f"Kinematic variable {kin_var} not found in NUISANCEEvent.")
            # print(f"Generating graph: {name} using {kin_var} = {kin_val}. Npts={npts}, range=({range_min}, {range_max}), prior={prior}")
            Spline_TGraph = Spline_obj.generate_graph(kin_val, kin_key=kin_var, n_points=npts, range_min=range_min, range_max=range_max)
            # append to the corresponding list
            Splines_lists[name].append(Spline_TGraph)


        # fill lists!
        SelectedSample_list.append(sample)
        Pmu_list.append(Pmu_smeared)
        CosThetamu_list.append(CosThetamu_smeared)
        TruePmu_list.append(Pmu)
        TrueCosThetamu_list.append(CosThetamu)
        Reaction_list.append(event.Mode)
        Enu_list.append(Enu)


    pbar.close()

    f.close()
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Finished reading {max_events} entries in {elapsed_time}.")

    # dummy weights branch (all weights = 1.0)
    weights_list = [1.0] * len(SelectedSample_list)
    # isRHC branch
    isRHC_list = [0 if isFHC else 1] * len(SelectedSample_list)

    dictionary = {
        "SelectedSample": to_1d_numpy(SelectedSample_list, dtype=np.int32, int_fill=-1),
        "Pmu": to_1d_numpy(Pmu_list, dtype=np.float32, float_fill=-999.0),
        "CosThetamu": to_1d_numpy(CosThetamu_list, dtype=np.float32, float_fill=-999.0),
        "TruePmu": to_1d_numpy(TruePmu_list, dtype=np.float32, float_fill=-999.0),
        "TrueCosThetamu": to_1d_numpy(TrueCosThetamu_list, dtype=np.float32, float_fill=-999.0),
        "Reaction": to_1d_numpy(Reaction_list, dtype=np.int32, int_fill=-999),
        "Enu": to_1d_numpy(Enu_list, dtype=np.float32, float_fill=-999.0),
        "weight": to_1d_numpy(weights_list, dtype=np.float32, float_fill=1.0),
        "isRHC": to_1d_numpy(isRHC_list, dtype=np.int32, int_fill=0),
    }
    # append to dictionary the splines lists (special handling for TGraph branches)
    for name, spline_list in Splines_lists.items():
        dictionary.update({ name: to_1d_numpy(spline_list, dtype=ROOT.TGraph, int_fill=None) })
        print(f"Added spline branch with: {name}, filled with {len(spline_list)} TGraphs. TGraph name: {spline_list[0].GetName() if len(spline_list) > 0 else 'N/A'}")



        # "Spline_Enu": to_1d_numpy(Spline_Enu_list, dtype=ROOT.TGraph, int_fill=None),
        # "Spline_PP": to_1d_numpy(Spline_PP_list, dtype=ROOT.TGraph, int_fill=None),
        # "Spline_FSIstrength": to_1d_numpy(Spline_FSIstrength_list, dtype=ROOT.TGraph, int_fill=None),
        # "Spline_mirror_q0": to_1d_numpy(Spline_mirror_q0_list, dtype=ROOT.TGraph, int_fill=None),
        # "Spline_mirror_q3": to_1d_numpy(Spline_mirror_q3_list, dtype=ROOT.TGraph, int_fill=None),
        # "Spline_bound_W": to_1d_numpy(Spline_bound_W_list, dtype=ROOT.TGraph, int_fill=None),



    return dictionary


def write_dictionary_to_tree(dictionary, output_root_file):
    """
    Write the dictionary of lists to a ROOT TTree, using pyROOT (supports TGraphs).

    Parameters
    ----------
    dictionary : dict
        Dictionary of lists to write.
    output_root_file : str
        Path to the output ROOT file.
    """

    # create ROOT file and TTree
    root_file = ROOT.TFile(output_root_file, "RECREATE")
    tree = ROOT.TTree("sample_sum", "GUNDAM input dataset from NUISANCE")

    # create branches
    normal_branches = {}
    special_branches = {}
    for key, values in dictionary.items():
        if len(values) == 0:
            print(f"Warning: empty values for key {key}, skipping branch creation.")
            continue

        if isinstance(values[0], ROOT.TGraph):
            # special handling for TGraph branch
            special_branches[key] = ROOT.TClonesArray("TGraph")
            tree.Branch(key, special_branches[key], 32000, 0)
        else:
            # normal branches
            arr = np.array(values)
            # determine type
            dtype = arr.dtype
            if np.issubdtype(dtype, np.integer):
                buf = np.zeros(1, dtype=np.int32)
                tree.Branch(key, buf, f"{key}/I")
            elif np.issubdtype(dtype, np.floating):
                buf = np.zeros(1, dtype=np.float32)
                tree.Branch(key, buf, f"{key}/F")
            else:
                raise ValueError(f"Unsupported dtype {dtype} for key {key}")
            normal_branches[key] = buf

    num_entries = len(next(iter(dictionary.values())))
    # fill tree (intantiate a progress bar)
    pbar = tqdm(total=num_entries)
    for i in range(num_entries):
        # fill numeric branches
        for key, buf in normal_branches.items():
            buf[0] = dictionary[key][i]

        # fill TGraph branches
        for key, tca in special_branches.items():
            tca.Clear("C")
            g = dictionary[key][i]
            # placement new into TClonesArray
            ROOT.TClonesArray(tca).ConstructedAt(0)
            tca[0] = g  # PyROOT magic works here

        tree.Fill()
        # progress bar
        if (i + 1) % (num_entries // 50) == 0:
            pbar.update(num_entries // 50)

    pbar.close()


    root_file.Write()
    root_file.Close()
    print(f"Wrote output ROOT file: {output_root_file}")


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Produce a GUNDAM input dataset from a NUISANCE flat tree.")
    p.add_argument("--file_path", "-f", type=str, help="Path to the NUISANCE flat tree ROOT file.")
    p.add_argument("--isFHC", type=int, nargs="?", default=None, help="1 if FHC beam mode, 0 if RHC. If not provided, inferred from file name.")
    p.add_argument("--max_events", type=int, default=-1, help="Maximum number of events to process. Default is -1 (all events).")
    p.add_argument("--output_plots_dir", type=str, default="./plots", help="Directory to save output plots. Default is ./plots")
    p.add_argument("--config_spline", "-c", required=True, type=str, help="Path to the SPLINE configuration YAML file.")
    p.add_argument("--ignore_cov_gen", action="store_true", help="If set, ignore the generate_cov_matrix option in the config file.")
    args = p.parse_args()

    isFHC = args.isFHC
    file_path = args.file_path
    max_events = args.max_events
    out_dir = args.output_plots_dir
    ignore_cov = args.ignore_cov_gen

    default_tree_name = "FlatTree_VARS"

    # load splines from config_spline file
    config_file = args.config_spline


    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # top-level options
    generate_cov = cfg.get("generate_cov_matrix", False)

    # list of splines
    splines_dictionary = cfg.get("splines", [])

    print("generate_cov_matrix =", generate_cov)
    print(f"Will produce a dataset with {len(splines_dictionary)} splines:")
    print()

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

        # get the first entry of the NUISANCEFile to check if the kin_variable exists
        f = NUISANCEFile(file_path, default_tree_name, relevant_keys=None, step=10000)
        event_test = next(iter(f))
        if not hasattr(event_test, kin_var):
                raise ValueError(f"Kinematic variable {kin_var} not found in NUISANCEEvent class.")

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


    if isFHC is None:
        if "_FHC_" in file_path:
            isFHC = True
        elif "_RHC_" in file_path:
            isFHC = False
        else:
            raise ValueError("Cannot determine beam mode (FHC or RHC) from file path.")

    print(f"Processing file: {file_path},\nisFHC={isFHC},\nmax_events={max_events},\nPlots will be saved to: {out_dir}")

    data = make_dataset_from_nuisance(file_path, isFHC, max_events=max_events, splines_dict=splines_dictionary, tree_name=default_tree_name)

    # create plots (unpack lists from dictionary)
    SelectedSample_list = data["SelectedSample"]
    Pmu_list = data["Pmu"]
    CosThetamu_list = data["CosThetamu"]
    Reaction_list = data["Reaction"]
    plot_utils.plot_samples(SelectedSample_list, Pmu_list, CosThetamu_list, Reaction_list, out_dir=out_dir)

    output_root_file = file_path.replace("_NUISFLAT.root", "_GUNDAMINPUT.root")
    write_dictionary_to_tree(data, output_root_file)

    generate_cov = generate_cov and (not ignore_cov)
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



