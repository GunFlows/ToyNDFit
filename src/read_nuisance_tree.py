
from utils import *
import plot_utils

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
    pmu_map = uproot.open("reso_maps/pmu_resolution_map.root")["pmu_resolution_map"]
    cthetamap = uproot.open("reso_maps/costhetamu_resolution_map.root")["costhetamu_resolution_map"]

    start_time = datetime.datetime.now()
    f = NUISANCEFile(file_path, tree_name, relevant_keys=None, step=10000)
    num_entries = len(f)

    # OUTPUT BRANCHES LISTS
    SelectedSample_list = []
    Pmu_list = []
    CosThetamu_list = []
    Reaction_list = []

    pbar = tqdm(total=100)
    for i, event in enumerate(f):
        # limit number of events for testing
        if i > max_events and max_events > 0:
            break
        # progress bar
        if (i + 1) % (num_entries // 100) == 0:
            pbar.update(1)
        # Perform selection
        sample = event.PerformSelection(isFHC)

        # compute kinematics and variables
        Pmu = event.Pmu()
        CosThetamu = event.CosThetamu()
        # smear Pmu, CosThetamu
        Pmu_smeared, CosThetamu_smeared = smear(Pmu, CosThetamu, pmu_map, cthetamap)

        # fill lists!
        SelectedSample_list.append(sample)
        Pmu_list.append(Pmu_smeared)
        CosThetamu_list.append(CosThetamu_smeared)
        Reaction_list.append(event.Mode)

    pbar.close()

    f.close()
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Finished reading {num_entries} entries in {elapsed_time}.")

    dictionary = {
        "SelectedSample": to_1d_numpy(SelectedSample_list, dtype=np.int32, int_fill=-1),
        "Pmu": to_1d_numpy(Pmu_list, dtype=np.float32, float_fill=-999.0),
        "CosThetamu": to_1d_numpy(CosThetamu_list, dtype=np.float32, float_fill=-999.0),
        "Reaction": to_1d_numpy(Reaction_list, dtype=np.int32, int_fill=-999),
    }


    return dictionary


if __name__ == "__main__":

    isFHC = None
    out_dir = "./plots"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if len(sys.argv) > 2:
            isFHC = bool(int(sys.argv[2]))
    else:
        print("Usage: python read_nuisance_tree.py <path_to_nuisance_root_file> <is_FHC>")
        print("No file path provided, using default file.")
        # default files
        file_path = "/Users/lorenzo/T2K-uniGe/NDFit_Gunflows/ToyNDFit/DATA/T2KND_FHC_numu_C8H8_NEUT580_1M_0000_NUISFLAT.root"
        # file_path = "/Users/lorenzo/T2K-uniGe/NDFit_Gunflows/ToyNDFit/DATA/T2KND_RHC_numubar_C8H8_NEUT580_1M_0000_NUISFLAT.root"

    if isFHC is None:
        if "_FHC_" in file_path:
            isFHC = True
        elif "_RHC_" in file_path:
            isFHC = False
        else:
            raise ValueError("Cannot determine beam mode (FHC or RHC) from file path.")

    print(f"Processing file: {file_path},\nisFHC={isFHC}")

    data = make_dataset_from_nuisance(file_path, isFHC, max_events=-1)

    SelectedSample_list = data["SelectedSample"]
    Pmu_list = data["Pmu"]
    CosThetamu_list = data["CosThetamu"]
    Reaction_list = data["Reaction"]

    plot_utils.plot_samples(SelectedSample_list, Pmu_list, CosThetamu_list, Reaction_list, out_dir=out_dir)

    # finally, create a root file with a TTree containing the smeared kinematics and SelectedSample branch
    output_root_file = file_path.replace("_NUISFLAT.root", "_GUNDAMINPUT.root")
    with uproot.recreate(output_root_file) as f_out:
        f_out.mktree("sample_sum",data)

        print(f"Created output ROOT file: {output_root_file}")
        print(f"Branches: {list(data.keys())}")
        print(f"Created plots in {out_dir} directory.")
