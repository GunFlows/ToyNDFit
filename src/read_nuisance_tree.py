import uproot
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from neut_codes import neut_reactions, neut_compact, neut_group
from src.nuisance_event import NUISANCEEvent, NUISANCEFile, SampleCodes

# Function to read the NUISANCE flat trees, smear the momenta based on a realistic momentum and angle resolution, and
# generates a "GUNDAM dataset": basically a root file with a TTree containing FS kinematics.
# One branch of the tree is "SelectedSample": a simple signal selection should be performed.



# Minimal version: produce only branches "SelectedSample", "Pmu", "CosThetamu".



def read_nuisance_tree(file_path, isFHC = True, max_events = -1, tree_name="FlatTree_VARS"):
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
    SelectedSample_list = []
    Pmu_list = []
    CosThetamu_list = []
    Reaction_list = []

    pbar = tqdm(total=1000)
    for i, event in enumerate(f):
        if i > max_events and max_events > 0:
            break
        if (i + 1) % (num_entries // 1000) == 0:
            pbar.update(1)
        # if (i + 1) % 20000 == 0:
        #     print(f"Read {i + 1}/{num_entries} entries...")
        Pmu = event.Pmu()
        CosThetamu = event.CosThetamu()

        sample = event.PerformSelection(isFHC)
        SelectedSample_list.append(sample)
        Pmu_list.append(Pmu)
        CosThetamu_list.append(CosThetamu)
        Reaction_list.append(event.Mode)
        # print(f"Event {i + 1}: SelectedSample = {sample}: {SampleCodes[sample]}\tPmu = {Pmu} GeV/c")
        # if Pmu is not None:
        #     print(f"Pmu = {event.Pmu()} GeV/c")
        # else:
        #     print("No muon found in this event.")

        # if sample == -1:
        #     vector_pmu = event.vectorPmu()
        #     if vector_pmu is not None:
        #         print(f"Event {i + 1}: No sample selected. {event.Mode}, {event.cc}, {event.PDGnu}, Pmu={Pmu:.3f}, CosThetamu={CosThetamu:.3f} Px={vector_pmu[0]}, Py={vector_pmu[1]}, Pz={vector_pmu[2]}")
        #     else:
        #         print(f"Event {i + 1}: No sample selected. {event.Mode}, {event.cc}, {event.PDGnu}, no muon here. ")

    pbar.close()

    f.close()
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Finished reading {num_entries} entries in {elapsed_time}.")


    return SelectedSample_list, Pmu_list, CosThetamu_list, Reaction_list

def plot_samples(SelectedSample_list, Pmu_list, CosThetamu_list, Reaction_list, out_dir="./"):
    """
    Plot the kinematic distributions for each sample in SelectedSample_list.
    """
    print("Plotting kinematic distributions...")

    os.makedirs(out_dir, exist_ok=True)
    unique, counts = np.unique(SelectedSample_list, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('SelectedSample Code')
    plt.ylabel('Number of Events')
    plt.title('Distribution of SelectedSample Codes')
    plt.xticks(unique)
    plt.savefig(f'{out_dir}/SelectedSample_distribution.png')
    plt.clf()

    for sample_code in unique:
        indices = [i for i, x in enumerate(SelectedSample_list) if x == sample_code]
        Pmu_sample = [Pmu_list[i] for i in indices if Pmu_list[i] is not None]
        CosThetamu_sample = [CosThetamu_list[i] for i in indices if CosThetamu_list[i] is not None]
        # stack histograms for different Reaction_list within the sample_code
        reaction_vals = [Reaction_list[i] for i in indices]

        # Group Pmu by reaction for this sample
        reaction_vals = [Reaction_list[i] for i in indices]
        Pmu_vals = [Pmu_list[i] for i in indices]
        CosThetamu_vals = [CosThetamu_list[i] for i in indices]

        unique_reactions, _ = np.unique(reaction_vals, return_counts=True)
        stacked_data = []
        labels = []
        for r in unique_reactions:
            pmus_for_r = [p for p, rv in zip(Pmu_vals, reaction_vals) if rv == r and p is not None]
            if len(pmus_for_r) > 0:
                stacked_data.append(pmus_for_r)
                reaction_dict = neut_group(r)
                labels.append(f"{reaction_dict['interaction_type']}, {reaction_dict['subcategory']}")

        if len(stacked_data) == 0:
            # nothing to plot for this sample
            continue

        plt.hist(stacked_data, bins=50, stacked=True, alpha=0.8, label=labels)
        plt.xlabel('Pmu (GeV/c)')
        plt.ylabel('Number of Events')
        plt.title(f'{SampleCodes[sample_code]} ({sum(len(d) for d in stacked_data)} events)')
        plt.legend(title='Reactions', fontsize='small')
        plt.savefig(f'{out_dir}/Pmu_distribution_sample_{sample_code}.png')
        plt.clf()

        stacked_data_ctheta = []
        labels_ctheta = []
        for r in unique_reactions:
            cthetas_for_r = [ct for ct, rv in zip(CosThetamu_vals, reaction_vals) if rv == r and ct is not None]
            if len(cthetas_for_r) > 0:
                stacked_data_ctheta.append(cthetas_for_r)
                reaction_dict = neut_group(r)
                labels_ctheta.append(f"{reaction_dict['interaction_type']}, {reaction_dict['subcategory']}")

        if len(stacked_data_ctheta) == 0:
            # nothing to plot for this sample
            continue

        plt.hist(stacked_data_ctheta, bins=50, stacked=True, alpha=0.8, label=labels_ctheta)
        plt.xlabel('CosThetamu')
        plt.ylabel('Number of Events')
        plt.title(f'{SampleCodes[sample_code]} ({sum(len(d) for d in stacked_data_ctheta)} events)')
        plt.legend(title='Reactions', fontsize='small')
        plt.savefig(f'{out_dir}/CosThetamu_distribution_sample_{sample_code}.png')
        plt.clf()


def plot_nu_energy(file_path, tree_name="FlatTree_VARS"):
    """
    Plot the neutrino energy distribution from the NUISANCE flat tree.
    """
    print("Plotting neutrino energy distribution...")

    f = NUISANCEFile(file_path, tree_name, relevant_keys=None, step=10000)
    num_entries = len(f)
    Px_nu_list = []
    Py_nu_list = []
    Pz_nu_list = []

    pbar = tqdm(total=1000)
    for i, event in enumerate(f):
        if (i + 1) % (num_entries // 1000) == 0:
            pbar.update(1)
        # if (i + 1) % 20000 == 0:
        #     print(f"Read {i + 1}/{num_entries} entries...")
        nu_mask = (np.array(event.pdg_init) == 14)
        if np.any(nu_mask):
            Px_nu = np.array(event.px_init)[nu_mask]
            Py_nu = np.array(event.py_init)[nu_mask]
            Pz_nu = np.array(event.pz_init)[nu_mask]
            if (len(Px_nu) > 1):
                print(f"Found {len(Px_nu)} neutrinos in event {i + 1}")
            elif(len(Px_nu) == 1):
                Px_nu_list.append(Px_nu[0])
                Py_nu_list.append(Py_nu[0])
                Pz_nu_list.append(Pz_nu[0])


    pbar.close()

    # draw histograms for Px_nu, Py_nu, Pz_nu
    plt.hist(Px_nu_list, bins=50, alpha=0.7)
    plt.xlabel('Px_nu (GeV/c)')
    plt.ylabel('Number of Events')
    plt.title('Neutrino Px Distribution')
    plt.savefig('test/Px_nu_distribution.png')
    plt.clf()
    plt.hist(Py_nu_list, bins=50, alpha=0.7)
    plt.xlabel('Py_nu (GeV/c)')
    plt.ylabel('Number of Events')
    plt.title('Neutrino Py Distribution')
    plt.savefig('test/Py_nu_distribution.png')
    plt.clf()
    plt.hist(Pz_nu_list, bins=50, alpha=0.7)
    plt.xlabel('Pz_nu (GeV/c)')
    plt.ylabel('Number of Events')
    plt.title('Neutrino Pz Distribution')
    plt.savefig('test/Pz_nu_distribution.png')
    plt.clf()



if __name__ == "__main__":
    # Example usage
    file_path = "/Users/lorenzo/T2K-uniGe/NDFit_Gunflows/ToyNDFit/DATA/T2KND_FHC_numu_C8H8_NEUT580_1M_0000_NUISFLAT.root"
    # file_path = "/Users/lorenzo/T2K-uniGe/NDFit_Gunflows/ToyNDFit/DATA/T2KND_RHC_numubar_C8H8_NEUT580_1M_0000_NUISFLAT.root"

    if "_FHC_" in file_path:
        isFHC = True
    elif "_RHC_" in file_path:
        isFHC = False
    else:
        raise ValueError("Cannot determine beam mode (FHC or RHC) from file path.")

    # plot_nu_energy(file_path)
    SelectedSample_list, Pmu_list, CosThetamu_list, Reaction_list = read_nuisance_tree(file_path, isFHC)
    plot_samples(SelectedSample_list, Pmu_list, CosThetamu_list, Reaction_list, out_dir="./plots")