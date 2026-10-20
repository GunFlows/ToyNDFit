#!/bin/bash

# Detector systematics for the extended likelihood: a finer (Pmu, CosThetamu) grid than PmuThetamuBins.txt, shared by
# all samples, with equal-statistics edges, and its prior from generate_prior_matrix.py (same generator as lin100.root).
# Usage: source setup.sh; bash scripts/produce_detector_prior_matrix.sh N_PMU N_COS

# throw an error if SOURCE_FOLDER is not set
if [ -z "${SOURCE_FOLDER}" ]; then
  echo "Error: SOURCE_FOLDER is not set. Please run setup.sh."
  exit 1
fi

n_pmu=${1:-30}
n_cos=${2:-30}
size=$((n_pmu * n_cos))
scale=0.06 # overall scale: 3-6 % per bin
seed=44
blobs=4

linear_folder=${BASE_FOLDER}/GundamWorkspace/inputs/parameters/linear

python3 ${SOURCE_FOLDER}/make_pmu_thetamu_binning.py \
  --input_files ${DATASET_FOLDER}/T2KND_FHC_numu_C8H8_NEUT580_1M_0000_GUNDAMINPUT.root ${DATASET_FOLDER}/T2KND_RHC_numubar_C8H8_NEUT580_1M_0000_GUNDAMINPUT.root \
  --n_pmu ${n_pmu} --n_cos ${n_cos} --output_file ${linear_folder}/PmuThetamuBins_${n_pmu}x${n_cos}.txt

python3 ${SOURCE_FOLDER}/generate_prior_matrix.py --output_file ${linear_folder}/prior_detector_${size}.root \
  --size ${size} --scale ${scale} --seed ${seed} --blobs ${blobs}
