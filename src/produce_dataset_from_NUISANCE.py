
from utils import *
import plot_utils
import argparse
from spline_generator import SplineGenerator
import ROOT

# Function to read the NUISANCE flat trees, smear the momenta based on a realistic momentum and angle resolution, and
# generates a "GUNDAM dataset": basically a root file with a TTree containing FS kinematics.
# One branch of the tree is "SelectedSample": a simple signal selection should be performed.



# Minimal version: produce only branches "SelectedSample", "Pmu", "CosThetamu".



def make_dataset_from_nuisance(file_path, isFHC = True, max_events = -1, tree_name="FlatTree_VARS"):
    """
    Read a NUISANCE flat tree and return a list of NUISANCEEvent objects.
    This is just a test function!

    Parameters
    ----------
    file_path : str
        Path to the ROOT file containing the NUISANCE flat tree.
    tree_name : str
        Name of the tree to read (default is "flat_tree").

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
    # Splines (to be saved as TGraphs in the output tree)
    Spline_Enu_list = []
    Spline_PP_list = []
    Spline_FSIstrength_list = []
    Spline_mirror_q0_list = []
    Spline_mirror_q3_list = []
    Spline_bound_W_list = []

    # instantiate spline generators
    Spline_Enu = SplineGenerator(spline_type="polynomial", power=3, range_width=-0.3, kin_min=0.0, kin_max=5) # enhance events with low neutrino energy
    Spline_PP = SplineGenerator(spline_type="polynomial", power=3, range_width=0.3, kin_min=0, kin_max=1) # enhance/reduce pion production events as a function of pion energy
    Spline_FSIstrength = SplineGenerator(spline_type="polynomial", power=3, range_width=0.3, kin_min=-0.2, kin_max=0.2) # reweight FSI strength
    Spline_mirror_q0 = SplineGenerator(spline_type="mirrored", power=3, range_width=0.3, kin_min=0, kin_max=0.5, min_fixer = 1) # reweight events based on  q0
    Spline_mirror_q3 = SplineGenerator(spline_type="mirrored", power=3, range_width=0.3, kin_min=0, kin_max=0.5, min_fixer=1) # reweight events based on  q3
    Spline_bound_W = SplineGenerator(spline_type="polynomial", power=1, range_width=0.3, kin_min=0, kin_max=1.5) # reweight events based on  W

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

        Spline_Enu_TGraph = Spline_Enu.generate_graph(Enu, kin_key="Enu")
        Spline_FSIstrength_TGraph = Spline_FSIstrength.generate_graph(event.FSI_strength(), kin_key="FSI_strength")
        if isPP:
            Spline_PP_TGraph = Spline_PP.generate_graph(pion_energy, kin_key="pion_energy")
        else:
            Spline_PP_TGraph = Spline_PP.empty_graph()

        Spline_mirror_q0_TGraph = Spline_mirror_q0.generate_graph(event.q0, kin_key="mirror_q0", n_points=30, range_min=-5, range_max=5)
        Spline_mirror_q3_TGraph = Spline_mirror_q3.generate_graph(event.q3, kin_key="mirror_q3", n_points=30, range_min=-5, range_max=5)
        Spline_bound_W_TGraph = Spline_bound_W.generate_graph(event.W, kin_key="W", n_points=2, range_min=0, range_max=3)

        # fill lists!
        SelectedSample_list.append(sample)
        Pmu_list.append(Pmu_smeared)
        CosThetamu_list.append(CosThetamu_smeared)
        TruePmu_list.append(Pmu)
        TrueCosThetamu_list.append(CosThetamu)
        Reaction_list.append(event.Mode)
        Enu_list.append(Enu)

        Spline_Enu_list.append(Spline_Enu_TGraph)
        Spline_PP_list.append(Spline_PP_TGraph)
        Spline_FSIstrength_list.append(Spline_FSIstrength_TGraph)
        Spline_mirror_q0_list.append(Spline_mirror_q0_TGraph)
        Spline_mirror_q3_list.append(Spline_mirror_q3_TGraph)
        Spline_bound_W_list.append(Spline_bound_W_TGraph)

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
        # special handling for TGraph branches
        "Spline_Enu": to_1d_numpy(Spline_Enu_list, dtype=ROOT.TGraph, int_fill=None),
        "Spline_PP": to_1d_numpy(Spline_PP_list, dtype=ROOT.TGraph, int_fill=None),
        "Spline_FSIstrength": to_1d_numpy(Spline_FSIstrength_list, dtype=ROOT.TGraph, int_fill=None),
        "Spline_mirror_q0": to_1d_numpy(Spline_mirror_q0_list, dtype=ROOT.TGraph, int_fill=None),
        "Spline_mirror_q3": to_1d_numpy(Spline_mirror_q3_list, dtype=ROOT.TGraph, int_fill=None),
        "Spline_bound_W": to_1d_numpy(Spline_bound_W_list, dtype=ROOT.TGraph, int_fill=None),
    }


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
    args = p.parse_args()

    isFHC = args.isFHC
    file_path = args.file_path
    max_events = args.max_events
    out_dir = args.output_plots_dir

    if isFHC is None:
        if "_FHC_" in file_path:
            isFHC = True
        elif "_RHC_" in file_path:
            isFHC = False
        else:
            raise ValueError("Cannot determine beam mode (FHC or RHC) from file path.")

    print(f"Processing file: {file_path},\nisFHC={isFHC},\nmax_events={max_events},\nPlots will be saved to: {out_dir}")

    data = make_dataset_from_nuisance(file_path, isFHC, max_events=max_events)

    # create plots (unpack lists from dictionary)
    SelectedSample_list = data["SelectedSample"]
    Pmu_list = data["Pmu"]
    CosThetamu_list = data["CosThetamu"]
    Reaction_list = data["Reaction"]
    plot_utils.plot_samples(SelectedSample_list, Pmu_list, CosThetamu_list, Reaction_list, out_dir=out_dir)

    output_root_file = file_path.replace("_NUISFLAT.root", "_GUNDAMINPUT.root")
    write_dictionary_to_tree(data, output_root_file)




    #
    # # finally, create a root file with a TTree containing all the information in the dictionary
    # output_root_file = file_path.replace("_NUISFLAT.root", "_GUNDAMINPUT.root")
    # with uproot.recreate(output_root_file) as f_out:
    #     f_out.mktree("sample_sum",data)
    #
    #     print(f"Created output ROOT file: {output_root_file}")
    #     print(f"Branches: {list(data.keys())}")
    #     print(f"Created plots in {out_dir} directory.")
