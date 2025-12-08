#!/bin/bash

executable="${SOURCE_FOLDER}/generate_prior_matrix.py"
output_file="${BASE_FOLDER}/GundamWorkspace/inputs/parameters/linear/prior_matrix.root"
scale=0.0001 # overall variance scale
seed=43

python3 ${executable} --output_file ${output_file} --scale ${scale} --seed ${seed}