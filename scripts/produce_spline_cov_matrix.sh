#!/bin/bash

# throw an error if SOURCE_FOLDER is not set
if [ -z "${SOURCE_FOLDER}" ]; then
  echo "Error: SOURCE_FOLDER is not set. Please run setup.sh."
  return 1
fi

executable=${SOURCE_FOLDER}/produce_spline_covariance_matrix.py

output_plots_dir=${DATASET_FOLDER}/plots_FHC
mkdir -p ${output_plots_dir}

python ${executable} --config_spline ${BASE_FOLDER}/config/config_splines.yaml

