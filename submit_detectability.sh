#!/bin/bash
#PBS -N predict-detect
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -o logs/detectability.out
#PBS -e logs/detectability.err
#PBS -m abe
#
# D036 detectability floor (decisions/D036-detectability-floor.md).
# Resumable: finished cells are skipped on resubmission.
#
#   mkdir -p logs && qsub submit_detectability.sh
#   mkdir -p logs && qsub -v ROOT=outputs_gates/d014 submit_detectability.sh
#   mkdir -p logs && qsub -v AMEND=1 submit_detectability.sh   # D036.1
#
# No --n-jobs flag: workers come from NCPUS via default_n_jobs().

set -uo pipefail
cd "$PBS_O_WORKDIR"

export PREDICT_CONDA_ROOT="$HOME/projects/miniconda3"
source "$PREDICT_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate predict_env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

ROOT="${ROOT:-outputs}"
AMEND="${AMEND:-0}"
EXTRA=""
[[ "$AMEND" == "1" ]] && EXTRA="--amendment"

echo "Job     : ${PBS_JOBID:-interactive}   NCPUS: ${NCPUS:-unset}"
echo "Workers : $(python -c 'from predict.config import default_n_jobs; print(default_n_jobs())')"
echo "Root    : $ROOT   amendment D036.1: $AMEND"
echo "Started : $(date -Is)"

python scripts/probe_detectability_floor.py --root "$ROOT" $EXTRA
rc=$?
echo "Finished: $(date -Is)  exit $rc"
exit $rc
