#!/bin/bash

# throw an error if SOURCE_FOLDER is not set
if [ -z "${SOURCE_FOLDER}" ]; then
  echo "Error: SOURCE_FOLDER is not set. Please run setup.sh."
  exit 1
fi

executable=${SOURCE_FOLDER}/produce_dataset_from_NUISANCE.py

input_nuisance_file="${DATASET_FOLDER}/T2KND_FHC_numu_C8H8_NEUT580_1M_0000_NUISFLAT.root"

python ${executable} \
      --file_path ${input_nuisance_file} \
      --isFHC 1


