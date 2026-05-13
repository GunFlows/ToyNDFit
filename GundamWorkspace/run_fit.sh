#!/bin/bash

# RUN THIS INSIDE CONTAINER!

ARGS="$@"

if command -v gundamFitter >/dev/null 2>&1; then
	echo "Fitter app found!"
else
	source /workspace/work/GuNFlows/demonstrator_scripts/setup.sh	
fi



gundamFitter ${ARGS}
