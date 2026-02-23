### This will be a README one day

Overview
--------
Tools to produce GUNDAM input datasets from NUISANCE flat trees and to generate related prior/covariance matrices and plots.

Requirements
------------
This project was assessed from the Python sources in `src/` and the configuration files in `config/`.
Below are the inferred runtime and development requirements and recommended setup steps.

1) Supported OS and Python
- Recommended: macOS or Linux (development done on macOS).  
- Python: 3.8+ (3.9 or 3.10 recommended). The code uses modern Python 3 features and Python numeric/scientific libraries.

2) System / non-Python dependencies
- CERN ROOT with PyROOT (required for full functionality: writing TGraph/TFile, TMatrixT, TVectorD, TCanvas, etc.). Many scripts import `ROOT` (pyROOT) and use ROOT classes.  
  - Recommended install (conda):
    conda create -n ndfit python=3.9
    conda activate ndfit
    conda install -c conda-forge root
  - On macOS you can also install ROOT via Homebrew or a binary distribution, but PyROOT must be available to the same Python interpreter.
- If you cannot install ROOT, many input-reading operations use `uproot` (pure Python) and can run in read-only mode, but writing output ROOT files with TGraphs and covariance generation requires PyROOT.
- A C/C++ compiler (gcc/clang) may be needed if you build auxiliary C/C++ code in `auxiliary/` or `scripts/`.

3) Python packages (see `requirements.txt`)
- numpy
- scipy
- matplotlib
- uproot
- tqdm

[//]: # (- pyyaml)

4) Environment variables and helper script
- `setup.sh` defines project paths that several scripts expect. Source it before running the Python tools:

  source setup.sh

  This sets (example):
  - BASE_FOLDER
  - SOURCE_FOLDER
  - DATASET_FOLDER
  - RESO_MAPS_FOLDER

  `produce_dataset_from_NUISANCE.py` expects `RESO_MAPS_FOLDER` to point to a folder containing `pmu_resolution_map.root` and `costhetamu_resolution_map.root`.

5) Typical installation steps (recommended)
- Create and activate a virtual environment (or conda env). Example (venv):

  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt

- Install ROOT / PyROOT separately (see point 2). If you installed ROOT into a Conda env, PyROOT will be available in that env.

6) Quick usage examples
- Source the paths and run the producer with a spline config (small example):

  source setup.sh
  python src/produce_dataset_from_NUISANCE.py \
    --file_path DATA/T2KND_FHC_numu_C8H8_NEUT580_1M_0000_NUISFLAT.root \
    --config_spline config/config_splines.yaml \
    --max_events 10000

- To generate a prior covariance matrix standalone (requires ROOT):

  python src/generate_prior_matrix.py -o output_prior.root --size 22 --filter 0.6

7) Important runtime notes & troubleshooting
- If you see "Can't find resolution maps because environment variable `RESO_MAPS_FOLDER` is not set", source `setup.sh` (or export `RESO_MAPS_FOLDER`) so `produce_dataset_from_NUISANCE.py` can find the resolution maps in `reso_maps/`.
- ImportError: `import ROOT` fails. This means PyROOT is not available in your Python environment. Install ROOT (recommended via conda-forge) and use the same Python interpreter as the project virtualenv.
- The code uses both `uproot` (reading) and `ROOT` (writing, plotting). For full read+write functionality install both.

8) Configs and data
- Spline configurations are in `config/config_splines.yaml` and are passed to the dataset production run via `--config_spline`. Ensure the `kin_variable` fields correspond to attributes/methods in `src/nuisance_event.py` (for example `Enu_true`, `total_pion_energy`, `FSI_strength`, ...).
- Example data files are under `DATA/` and example resolution maps under `reso_maps/`.

Further work / optional
----------------------
- Add a `requirements.txt` (contained in the repository) and a simple `Makefile` or `scripts/bootstrap.sh` to automate env creation and ROOT installation guidance.
- Add small unit tests to validate the main data flow (reading a small test root file and producing the output tree).
