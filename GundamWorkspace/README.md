# Gundam base folder
#### In this folder we keep all the configuration files used to run a Gundam fit, as well as its outputs. Use meaningful naming of configurations to distinguish different versions.

## 8 Dec 2025: First working config (configToyOA.yaml).

This first iteration of configuration and inputs performs a simple Gundam fit with 6 samples (3 in RHC, 3 in FHC) and 22 correlated parameters. 
The parameters are all "linear", mimicking the flux parameters or pre-2024 detSyst. parameters.
Both the Asimov and Toy fits provide sensible results (fit converges, Hesse is found). 

## Next steps:
Include xsec-like parameters that use spline transformations.