import uproot
import numpy as np

class NUISANCEEvent:
    """Container for one entry of a NUISANCE flat tree."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

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
        muon_mask = (pdg == 13)
        if not np.any(muon_mask):
            return None  # No muon found
        Px = np.array(self.px)[muon_mask]
        Py = np.array(self.py)[muon_mask]
        Pz = np.array(self.pz)[muon_mask]

        if len(Px) > 1:
            # If multiple muons, there is a problem
            raise ValueError("Multiple muons found in event.")

        return np.sqrt(Px[0]**2 + Py[0]**2 + Pz[0]**2)



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
