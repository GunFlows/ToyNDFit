#!/bin/bash
#SBATCH --job-name=mcmc_analysis
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/shares/sanchezf/gundam_n_flow/ToyNDFit_dev/logs/mcmc_analysis_%j.out
#SBATCH --error=/home/shares/sanchezf/gundam_n_flow/ToyNDFit_dev/logs/mcmc_analysis_%j.err

set -euo pipefail

module load apptainer 2>/dev/null || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

BASE="/home/shares/sanchezf/gundam_n_flow/ToyNDFit_dev"
HOST_OUT="${BASE}/GundamWorkspace/output"
HOST_LOGS="${BASE}/logs"
HOST_ANALYSIS="${BASE}/mcmc_analysis"

HOST_GUNFLOWS="/home/shares/sanchezf/gundam_n_flow/GuNFlows"
SIF="/home/shares/sanchezf/gundam_n_flow/GuNFlows/env/containers/ml_image2.sif"

IN_GUNFLOWS="/workspace/work/GuNFlows"
IN_SETUP="${IN_GUNFLOWS}/setup_nosubshell.sh"
IN_OUT="/workspace/output"
IN_ANALYSIS="/workspace/mcmc_analysis"

mkdir -p "${HOST_LOGS}"
mkdir -p "${HOST_ANALYSIS}/results"

INPUT_HOST="${1:-${HOST_OUT}/gundamFitter_configToyOA_With_configMcmcToy_Asimov.root}"
if [ ! -f "${INPUT_HOST}" ]; then
  echo "ERROR: input file not found: ${INPUT_HOST}"
  exit 1
fi

RES_HOST="${HOST_ANALYSIS}/results/job_${SLURM_JOB_ID}"
mkdir -p "${RES_HOST}"

echo "Job started at $(date)"
echo "Input ROOT : ${INPUT_HOST}"
echo "Results   : ${RES_HOST}"

srun --ntasks=1 apptainer exec \
  --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --bind "${HOST_GUNFLOWS}:${IN_GUNFLOWS}" \
  --bind "${HOST_OUT}:${IN_OUT}" \
  --bind "${HOST_ANALYSIS}:${IN_ANALYSIS}" \
  --pwd "${IN_ANALYSIS}" \
  "${SIF}" bash -lc "\
    set -euo pipefail; \
    source '${IN_SETUP}'; \
    python3 -u '${IN_ANALYSIS}/analyze_mcmc.py' \
      --input '${IN_OUT}/$(basename "${INPUT_HOST}")' \
      --outdir '${IN_ANALYSIS}/results/job_${SLURM_JOB_ID}' \
      --dims '20:25' \
      --thin 1 \
      --burnin-frac 0.2 \
  "

echo "Job ended at $(date)"
echo "Results at: ${RES_HOST}"
