import uproot
import numpy as np

class NUISANCEEvent:
    """
    Container for one entry of a NUISANCE flat tree.
    It can be built directly from uproot data for a given index.
    """

    def __init__(self, **kwargs):
        # store all fields dynamically
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_tree(cls, tree, index):
        """
        Create an event from a NUISANCE flat tree entry.

        Parameters
        ----------
        tree : uproot.behaviors.TBranch.TTree
            The opened uproot tree (e.g. file["flat_tree"])
        index : int
            Index of the event to read
        """
        data = {}
        for branch_name in tree.keys():
            arr = tree[branch_name].array(entry_start=index, entry_stop=index+1)
            # uproot returns awkward arrays; get the scalar
            if len(arr) > 0:
                val = arr[0]
                # If branch contains a jagged array (e.g. multiple particles)
                if hasattr(val, "__len__") and not isinstance(val, (bytes, str)):
                    val = np.array(val)
                data[branch_name] = val
        return cls(**data)

    def __repr__(self):
        # Compact but readable summary
        keys = list(self.__dict__.keys())
        sample = ", ".join(f"{k}={getattr(self,k)!r}" for k in keys[:5])
        if len(keys) > 5:
            sample += ", ..."
        return f"<NUISANCEEvent {sample}>"