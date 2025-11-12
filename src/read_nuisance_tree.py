import uproot
import datetime
from src.nuisance_event import NUISANCEEvent, NUISANCEFile

# Function to read the NUISANCE flat trees, smear the momenta based on a realistic momentum and angle resolution, and
# generates a "GUNDAM dataset": basically a root file with a TTree containing FS kinematics.
# One branch of the tree is "SelectedSample": a simple signal selection should be performed.



# Minimal version: produce only branches "SelectedSample", "Pmu", "CosThetamu".



def read_nuisance_tree(file_path, tree_name="FlatTree_VARS"):
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
    events : list of NUISANCEEvent
        List of NUISANCEEvent objects representing each entry in the tree.
    """
    start_time = datetime.datetime.now()
    f = NUISANCEFile(file_path, tree_name, relevant_keys=None, step=10000)
    num_entries = len(f)
    for i, event in enumerate(f):
        if (i + 1) % 100000 == 0:
            print(f"Read {i + 1}/{num_entries} entries...")
        Pmu = event.Pmu()
        if Pmu is not None:
            print(f"Pmu = {event.Pmu()} GeV/c")
        else:
            print("No muon found in this event.")
    f.close()
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Finished reading {num_entries} entries in {elapsed_time}.")





if __name__ == "__main__":
    # Example usage
    file_path = "/Users/lorenzo/T2K-uniGe/NDFit_Gunflows/ToyNDFit/DATA/T2KND_FHC_numu_C8H8_NEUT580_1M_0000_NUISFLAT.root"
    read_nuisance_tree(file_path)