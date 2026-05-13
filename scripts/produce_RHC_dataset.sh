#!/bin/bash

# throw an error if SOURCE_FOLDER is not set
if [ -z "${SOURCE_FOLDER}" ]; then
  echo "Error: SOURCE_FOLDER is not set. Please run setup.sh."
  return 1
fi

executable=${SOURCE_FOLDER}/produce_dataset_from_NUISANCE.py

input_nuisance_file="${DATASET_FOLDER}/T2KND_RHC_numubar_C8H8_NEUT580_1M_0000_NUISFLAT.root"
output_mc_file="${DATASET_FOLDER}/T2KND_RHC_numubar_C8H8_NEUT580_1M_0000_GUNDAMINPUT.root"
output_data_file="${DATASET_FOLDER}/T2KND_RHC_numubar_C8H8_NEUT580_1M_0000_GUNDAM_DATA.root"

output_plots_dir=${DATASET_FOLDER}/plots_RHC
mkdir -p ${output_plots_dir}

python ${executable} \
      --file_path ${input_nuisance_file} \
      --isFHC 0 \
      --output_plots_dir ${output_plots_dir} \
      --ignore_cov_gen \
      --config_spline ${BASE_FOLDER}/config/config_splines_100p10.yaml \
      --output_root_file_mc ${output_mc_file} \
      --output_root_file_data ${output_data_file} \
#      --max_events 1000 \
