#!/bin/bash

executable="${SOURCE_FOLDER}/generate_prior_matrix.py"
output_file="${BASE_FOLDER}/GundamWorkspace/inputs/parameters/linear/lin100.root"
scale=0.03 # overall variance scale
seed=43
size=100
#filter=0.0001
blobs=4

python3 ${executable} --output_file ${output_file} --scale ${scale} --seed ${seed} --size ${size} --blobs ${blobs}