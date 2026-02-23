import uproot
import numpy as np

class NUISANCEEvent:
    """Container for one entry of a NUISANCE flat tree."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.SelectedSample = -1  # Placeholder for SelectedSample attribute

        # muon acceptance
        self.pmu_threshold = 0.01  # GeV/c
        self.costhetamu_threshold = 0.5
        # piplus acceptance
        self.ppiplus_threshold = 0.2  # GeV/c
        self.costhetapiplus_threshold = 0.5
        # piminus acceptance
        self.ppiminus_threshold = 0.2  # GeV/c
        self.costhetapiminus_threshold = 0.5
        # proton acceptance
        self.pproton_threshold = 0.45  # GeV/c
        self.costhetaproton_threshold = 0.5


    def __repr__(self):
        keys = list(self.__dict__.keys())
        sample = ", ".join(f"{k}={getattr(self,k)!r}" for k in keys[:5])
        if len(keys) > 5:
            sample += ", ..."
        return f"<NUISANCEEvent {sample}>"

    def Pmu(self):
        """Calculate muon momentum from Px, Py, Pz."""
        # loop through the pdg attribute (is a list) to find the muon (pdg = 13)
        pdg = np.array(self.pdg)
        muon_mask = (pdg == 13) | (pdg == -13)
        if not np.any(muon_mask):
            return None  # No muon found
        Px = np.array(self.px)[muon_mask]
        Py = np.array(self.py)[muon_mask]
        Pz = np.array(self.pz)[muon_mask]

        if len(Px) > 1:
            # Take leading muon if multiple muons found
            max_index = np.argmax(np.sqrt(Px**2 + Py**2 + Pz**2))
            return np.sqrt(Px[max_index]**2 + Py[max_index]**2 + Pz[max_index]**2)

        return 1000 * np.sqrt(Px[0]**2 + Py[0]**2 + Pz[0]**2) # MeV/c!!!

    def CosThetamu(self):
        """Calculate muon cos(theta) from Px, Py, Pz."""
        pdg = np.array(self.pdg)
        muon_mask = (pdg == 13) | (pdg == -13)
        if not np.any(muon_mask):
            return None  # No muon found
        Px = np.array(self.px)[muon_mask]
        Py = np.array(self.py)[muon_mask]
        Pz = np.array(self.pz)[muon_mask]
        if len(Px) > 1:
            # If multiple muons, take leading muon
            max_index = np.argmax(np.sqrt(Px**2 + Py**2 + Pz**2))
            P = np.sqrt(Px[max_index]**2 + Py[max_index]**2 + Pz[max_index]**2)
            return Pz[max_index] / P

        P = np.sqrt(Px[0]**2 + Py[0]**2 + Pz[0]**2)
        return Pz[0] / P

    def magnitude_pmiss(self):
        """Calculate the magnitude of missing momentum."""
        px = self.pmiss.member("fX")
        py = self.pmiss.member("fY")
        pz = self.pmiss.member("fZ")
        pmiss = np.linalg.norm([px, py, pz])
        # Calculate components for debugging
        return pmiss

    def isPP(self):
        """Return 1 if the event is a pion production event, 0 otherwise."""
        pdg = np.array(self.pdg)
        pions_mask = (pdg == 211) | (pdg == -211) | (pdg == 111)
        if np.any(pions_mask):
            return 1
        else:
            return 0

    def total_pion_energy(self):
        """Calculate total energy of all pions in the event."""
        pdg = np.array(self.pdg)
        pions_mask = (pdg == 211) | (pdg == -211) | (pdg == 111)
        if not np.any(pions_mask):
            return 0.0
        E = np.array(self.E)[pions_mask]
        total_E = np.sum(E)
        return total_E

    def PNucleon_nuantinu(self):
        """ Return outgoing nucleon momentum, make it negative if antineutrino"""
        pdg = np.array(self.pdg)
        nucleon_mask = (pdg == 2212) | (pdg == -2212)
        if not np.any(nucleon_mask):
            return 0.0
        Px = np.array(self.px)[nucleon_mask]
        Py = np.array(self.py)[nucleon_mask]
        Pz = np.array(self.pz)[nucleon_mask]
        if len(Px) > 1:
            # If multiple nucleons, take leading nucleon
            max_index = np.argmax(np.sqrt(Px**2 + Py**2 + Pz**2))
            P = np.sqrt(Px[max_index]**2 + Py[max_index]**2 + Pz[max_index]**2)
            if self.PDGnu == 14:  # Neutrino case
                return P
            else:  # Antineutrino case
                return -P
        else:
            P = np.sqrt(Px[0]**2 + Py[0]**2 + Pz[0]**2)
            if self.PDGnu == 14:  # Neutrino case
                return P
            else:  # Antineutrino case
                return -P

    def FSI_strength(self):
        """Return the FSI strength of the event computed as energy of the escaping proton - energy of the proton at vertex."""

        pdg_vert = np.array(self.pdg_vert)
        proton_mask = (pdg_vert == 2212)
        pdg_out = np.array(self.pdg)
        proton_mask_out = (pdg_out == 2212)
        # print(f"N protons at vertex: {np.sum(proton_mask)}, N protons out: {np.sum(proton_mask_out)}")
        K_proton_vertex = np.sum(np.array(self.E_vert)[proton_mask]) - 0.93827*np.sum(proton_mask)
        K_proton_out = np.sum(np.array(self.E)[proton_mask_out]) - 0.93827*np.sum(proton_mask_out)
        # print(f"K_proton_vertex: {K_proton_vertex}, K_proton_out: {K_proton_out}")
        FSI_strength_value = K_proton_vertex - K_proton_out

        return FSI_strength_value


    def vectorPmu(self):
        """Return the muon momentum vector (Px, Py, Pz) as a list."""
        pdg = np.array(self.pdg)
        muon_mask = (pdg == 13) | (pdg == -13)
        if not np.any(muon_mask):
            return None  # No muon found
        Px = np.array(self.px)[muon_mask]
        Py = np.array(self.py)[muon_mask]
        Pz = np.array(self.pz)[muon_mask]
        if len(Px) > 1:
            # If multiple muons, take leading muon
            max_index = np.argmax(np.sqrt(Px**2 + Py**2 + Pz**2))
            return [Px[max_index], Py[max_index], Pz[max_index]]

        return [Px[0], Py[0], Pz[0]]

    def print_gamma_info(self):
        """Print information about gamma particles in the event."""
        pdg = np.array(self.pdg)
        gamma_mask = (pdg == 22)
        if not np.any(gamma_mask):
            # print("No gamma particles found in this event.")
            return
        Px = np.array(self.px)[gamma_mask]
        Py = np.array(self.py)[gamma_mask]
        Pz = np.array(self.pz)[gamma_mask]
        E = np.array(self.E)[gamma_mask]
        for i in range(len(Px)):
            P = np.sqrt(Px[i]**2 + Py[i]**2 + Pz[i]**2)
            print(f"Gamma {i+1}: E={E[i]}")

    def PerformSelection(self, isFHC):
        """Perform a simple signal selection based on presence of muon and pion

        Parameters
        ----------
        isFHC : bool
            True if the beam mode is FHC, False for RHC.

        Returns
        -------
        SelectedSample code : int

        """
        if isFHC:
            self.PerformSelectionFHC()
        else:
            self.PerformSelectionRHC()

        return self.SelectedSample

    def PerformSelectionFHC(self):
        """Perform selection for FHC mode."""
        pdg = np.array(self.pdg)
        has_muon = np.any(pdg == 13)
        has_antimuon = np.any(pdg == -13)
        if has_antimuon and has_muon:
            return -1  # Not selected: both muon and antimuon present

        if has_muon:
            muon_pmu = self.Pmu()
            muon_costhetamu = self.CosThetamu()
            has_muon = has_muon and (muon_pmu > self.pmu_threshold) and (muon_costhetamu > self.costhetamu_threshold)

        num_piplus = 0
        piplus_mask = (pdg == 211)
        if np.any(piplus_mask):
            Px_piplus = np.array(self.px)[piplus_mask]
            Py_piplus = np.array(self.py)[piplus_mask]
            Pz_piplus = np.array(self.pz)[piplus_mask]
            for i in range(len(Px_piplus)):
                P_piplus = np.sqrt(Px_piplus[i]**2 + Py_piplus[i]**2 + Pz_piplus[i]**2)
                costheta_piplus = Pz_piplus[i] / P_piplus
                if P_piplus > self.ppiplus_threshold and costheta_piplus > self.costhetapiplus_threshold:
                    num_piplus += 1

        if has_muon:
            if num_piplus == 0:
                self.SelectedSample = 1
            elif num_piplus == 1:
                self.SelectedSample = 2
            else:
                self.SelectedSample = 3
        else:
            self.SelectedSample = -1  # Not selected

    def PerformSelectionRHC(self):
        """Perform selection for RHC mode."""
        pdg = np.array(self.pdg)
        has_antimuon = np.any(pdg == -13)
        has_muon = np.any(pdg == 13)
        if has_antimuon and has_muon:
            return -1  # Not selected: both muon and antimuon present

        if has_muon:
            muon_pmu = self.Pmu()
            muon_costhetamu = self.CosThetamu()
            has_muon = has_muon and (muon_pmu > self.pmu_threshold) and (muon_costhetamu > self.costhetamu_threshold)

        if has_antimuon:
            antimuon_mask = (pdg == -13)
            Px_antimuon = np.array(self.px)[antimuon_mask]
            Py_antimuon = np.array(self.py)[antimuon_mask]
            Pz_antimuon = np.array(self.pz)[antimuon_mask]
            if len(Px_antimuon) > 1:
                raise ValueError("Multiple antimuons found in event.")
            P_antimuon = np.sqrt(Px_antimuon[0]**2 + Py_antimuon[0]**2 + Pz_antimuon[0]**2)
            costheta_antimuon = Pz_antimuon[0] / P_antimuon
            has_antimuon = has_antimuon and (P_antimuon > self.pmu_threshold) and (costheta_antimuon > self.costhetamu_threshold)

        num_piplus = 0
        piplus_mask = (pdg == 211)
        if np.any(piplus_mask):
            Px_piplus = np.array(self.px)[piplus_mask]
            Py_piplus = np.array(self.py)[piplus_mask]
            Pz_piplus = np.array(self.pz)[piplus_mask]
            for i in range(len(Px_piplus)):
                P_piplus = np.sqrt(Px_piplus[i]**2 + Py_piplus[i]**2 + Pz_piplus[i]**2)
                costheta_piplus = Pz_piplus[i] / P_piplus
                if P_piplus > self.ppiplus_threshold and costheta_piplus > self.costhetapiplus_threshold:
                    num_piplus += 1

        num_piminus = 0
        piminus_mask = (pdg == -211)
        if np.any(piminus_mask):
            Px_piminus = np.array(self.px)[piminus_mask]
            Py_piminus = np.array(self.py)[piminus_mask]
            Pz_piminus = np.array(self.pz)[piminus_mask]
            for i in range(len(Px_piminus)):
                P_piminus = np.sqrt(Px_piminus[i]**2 + Py_piminus[i]**2 + Pz_piminus[i]**2)
                costheta_piminus = Pz_piminus[i] / P_piminus
                if P_piminus > self.ppiminus_threshold and costheta_piminus > self.costhetapiminus_threshold:
                    num_piminus += 1

        if has_antimuon:
            if num_piminus == 0:
                self.SelectedSample = 59
            elif num_piminus == 1:
                self.SelectedSample = 60
            else:
                self.SelectedSample = 61
        elif has_muon:
            # Check for nu_mu background
            has_muon = np.any(pdg == 13)
            if has_muon:
                if num_piplus == 0:
                    self.SelectedSample = 71
                elif num_piplus == 1:
                    self.SelectedSample = 72
                else:
                    self.SelectedSample = 73
            else:
                self.SelectedSample = -1 # Not selected
        else:
            self.SelectedSample = -1  # Not selected








class NUISANCEFile:
    """Iterator over NUISANCE flat tree events."""
    def __init__(self, filepath, treename, relevant_keys, step=1000):
        """
        Parameters
        ----------
        filepath : str
            Path to the ROOT file
        treename : str
            Name of the TTree (usually 'flat_tree')
        step : int
            Number of entries to read per chunk
        """
        self.file = uproot.open(filepath)
        self.tree = self.file[treename]
        self.step = step
        if relevant_keys is None:
            self.keys = self.tree.keys()
        else:
            self.keys = relevant_keys
        self._nentries = self.tree.num_entries

    def __iter__(self):
        """Return an iterator over all events."""
        for arrays in self.tree.iterate(self.keys, step_size=self.step, library="np"):
            n = len(next(iter(arrays.values())))
            for i in range(n):
                data = {k: arrays[k][i] for k in self.keys}
                yield NUISANCEEvent(**data)

    def __len__(self):
        return self._nentries

    def close(self):
        self.file.close()


# Sample codes mapping
SampleCodes = {
    -1: "not selected",
    # "classic" FHC samples
    1: "numu CC 0pi in FHC",
    2: "numu CC 1pi in FHC",
    3: "numu CCother in FHC",
    # more detailed T2K samples (FHC)
    7:  "CC 0pi 0p 0gamma in FHC",
    8:  "CC 0pi Np 0gamma in FHC",
    10: "CC 1pi 0gamma in FHC",
    12: "CC other 0gamma in FHC",
    13: "CC other gamma in FHC",
    # RHC samples
    59: "antinumu CC 0pi in RHC",
    60: "antinumu CC 1pi in RHC",
    61: "antinumu CCother in RHC",
    71: "numu bkg CC 0pi in RHC",
    72: "numu bkg CC 1pi in RHC",
    73: "numu bkg CCother in RHC"
}


