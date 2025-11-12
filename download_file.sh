#!/bin/bash
filename="T2KND_RHC_numubar_C8H8_NEUT580_1M_0000_NUISFLAT.root"
url="https://portal.nersc.gov/project/nuisance/MC_IOP_review/NEUT580/${filename}"
destination_path="/Users/lorenzo/T2K-uniGe/NDFit_Gunflows/ToyNDFit/DATA"
output="${destination_path}/${filename}"

curl -L -o "$output" "$url"
