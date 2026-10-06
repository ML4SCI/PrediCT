#!/bin/bash
#PBS -N predict-pipeline
#PBS -l select=1:ncpus=16:mem=96gb
#PBS -l walltime=48:00:00
#PBS -q workq
#PBS -o logs/pipeline.out
#PBS -e logs/pipeline.err
#PBS -m abe
#
# PrediCT: PBS Pro submission wrapper.
#
#   qsub submit_pipeline.sh                                   # stages 1-10
#   qsub -v PIPELINE_ARGS="--from 1 --to 1" submit_pipeline.sh  # one stage
#   qsub -v PIPELINE_ARGS="--from 11 --to 11" submit_pipeline.sh # self-checks
#
# Run `mkdir -p logs` first or PBS cannot write the -o / -e files and the
# job fails before it starts.
#
# select=1 is deliberate. joblib parallelism here is shared-memory only,
# so a second node would sit idle for the whole walltime. Scale ncpus,
# not select.
#
# Parallelism is NOT hardcoded below. run_pipeline.sh and
# predict.config.default_n_jobs() both read PBS_NCPUS, so the -l select
# line above is the single place worker count is controlled. Change
# ncpus and everything downstream follows.

set -euo pipefail

cd "$PBS_O_WORKDIR"

# ── conda ─────────────────────────────────────────────────────────────
# PREDICT_CONDA_ROOT is checked first by run_pipeline.sh's search loop,
# so setting it here keeps the two scripts from disagreeing.
export PREDICT_CONDA_ROOT="$HOME/projects/miniconda3"
source "$PREDICT_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate predict_env

# ── BLAS thread control ───────────────────────────────────────────────
# Exported before any Python starts. run_pipeline.sh sets these too; both
# are needed because either script can be the entry point.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ── provenance ────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────"
echo "Job        : ${PBS_JOBID:-interactive}"
echo "Node       : $(hostname)"
echo "Workdir    : $PBS_O_WORKDIR"
echo "PBS_NCPUS  : ${PBS_NCPUS:-unset}   NCPUS: ${NCPUS:-unset}"
if [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]]; then
    # Diagnostic only. PBS Pro writes one line per CHUNK, not per core, so
    # this count is NOT the worker count and nothing reads it as such.
    echo "Nodefile   : $(grep -c . "$PBS_NODEFILE") chunk line(s) [not core count]"
fi
echo "Python     : $(which python)"
echo "Conda env  : ${CONDA_DEFAULT_ENV:-unset}"
# REVISION is written by the sync (the cluster copy has no current .git).
echo "Revision   : $(cat REVISION 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Stages     : ${PIPELINE_ARGS:-all (1-10)}"
echo "Started    : $(date -Is)"
echo "──────────────────────────────────────────────────────────"

# ── run ───────────────────────────────────────────────────────────────
# No --n-jobs flag: run_pipeline.sh resolves it from PBS_NCPUS itself.
# Passing it here would silently override the scheduler and reintroduce
# the mismatch this whole setup exists to prevent.
# shellcheck disable=SC2086
bash ./run_pipeline.sh ${PIPELINE_ARGS:-} "$@"   # bash: not marked executable in git

echo "──────────────────────────────────────────────────────────"
echo "Finished   : $(date -Is)"
echo "──────────────────────────────────────────────────────────"
