

# code to popen a ttree (using uproot), make a 2 histogram of Global_FrontMomentum - True_momentum vs True_momentum (branch names)
# then plot the resolution (RMS/mean) as a function of True_momentum
import uproot
import numpy as np
import matplotlib.pyplot as plt

def extract_momentum_resolution(root_file, tree_name='FlatTree'):
    """
    Extract and plot the momentum resolution from a NUISANCE root file.

    Parameters
    ----------
    root_file : str
        Path to the root file.
    tree_name : str
        Name of the TTree in the root file (default is 'nuisance_tree').
    """
    # Open the root file and access the tree
    with uproot.open(root_file) as file:
        tree = file[tree_name]

        # Read the relevant branches
        true_momentum = tree['True_momentum'].array(library='np')
        front_momentum = tree['Global_FrontMomentum'].array(library='np')

        # Calculate momentum difference
        momentum_diff = front_momentum - true_momentum
        mask = (front_momentum > 0) & (front_momentum < 5000)
        true_momentum = true_momentum[mask]
        front_momentum = front_momentum[mask]
        momentum_diff = momentum_diff[mask]

        # Define bins for true momentum
        bins = np.linspace(0, np.max(true_momentum), 50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Calculate resolution (RMS/mean) in each bin
        resolution = []
        for i in range(len(bins) - 1):
            mask = (true_momentum >= bins[i]) & (true_momentum < bins[i + 1])
            if np.sum(mask) > 0:
                rms = np.sqrt(np.mean(momentum_diff[mask]**2))
                print(rms)
                mean = bins[i] + (bins[i + 1] - bins[i]) / 2
                resolution.append(rms / mean )
            else:
                resolution.append(0)

        plt.figure(figsize=(10, 6))
        plt.hist(true_momentum, bins=50, color='blue', alpha=0.7, range=(0,5000))
        plt.xlabel('True Momentum (GeV/c)')
        plt.ylabel('Number of Events')
        plt.title('True Momentum Distribution')
        plt.grid()
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(front_momentum, bins=50, color='orange', alpha=0.7, range=(0,5000))
        plt.xlabel('Reconstructed Front Momentum (GeV/c)')
        plt.ylabel('Number of Events')
        plt.title('Reconstructed Front Momentum Distribution')
        plt.grid()
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist2d(true_momentum, momentum_diff, bins=50, cmap='viridis', range=[[0, 5000], [ -500, 500]])
        plt.xlabel('True Momentum (GeV/c)')
        plt.ylabel('Momentum Difference (GeV/c)')
        plt.title('Momentum Difference vs True Momentum')
        plt.colorbar(label='Counts')
        plt.show()
        plt.close()

        # Plot the resolution
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, resolution, marker='o', linestyle='-')
        plt.xlim(0,5000)
        plt.ylim(0,1)
        plt.xlabel('True Momentum (GeV/c)')
        plt.ylabel('Momentum Resolution (RMS/Mean)')
        plt.title('Momentum Resolution vs True Momentum')
        plt.grid()
        plt.show()



if __name__ == "__main__":
    root_file_path = 'FlatTree_prod8V10_FHC_run13c_partial.root'
    extract_momentum_resolution(root_file_path)