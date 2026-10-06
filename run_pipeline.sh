#!/usr/bin/env bash
#
# PrediCT: end-to-end pipeline runner.
#
# Runs the stages below in order. Each stage logs to its own file under
# outputs/_logs/, with a master log per run.
#
#   1  discovery            scripts/01_discover.py
#   2  preprocess           scripts/02_preprocess.py
#   3  features             scripts/03_features.py, 03b_cohort_sanity.py
#   4  stability gates      D014 (supplementary): 04_perturbations, 05_icc_gate
#                           D033 (primary): calibrate_realistic, 04b (all arms),
#                           05b gate; D034 tiers (05c, exploratory)
#   5  reduce x3 cohorts    reads the gate named by stability.primary_gate
#   6  discover x3 cohorts
#   7  analyse + reports
#   8  validate
#   9  lesion chain         lesion morphology (D037), k sweep, C8 tests, LAD,
#                           isolated LM, lesion stability (exploratory)
#  10  figures
#  11  self-checks          kernel diagnostics, detectability floor (D036.1);
#                           long, so only when asked for (--to 11 or --from 11)
#
# Usage:
#   ./run_pipeline.sh                              # stages 1-10
#   ./run_pipeline.sh --from 5                     # start at stage 5
#   ./run_pipeline.sh --from 5 --to 8              # 5 to 8 only
#   ./run_pipeline.sh --from 11 --to 11            # self-checks only
#   ./run_pipeline.sh --skip-exploratory           # skip stage 9
#   ./run_pipeline.sh --n-jobs 40                  # override parallelism
#   ./run_pipeline.sh --dry-run                    # print what would run
#
# Exits non-zero on the first failing stage so CI / cron can detect it.

set -euo pipefail

# ─────────────────────── defaults ───────────────────────

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parallelism, resolved from the batch scheduler ────────────────────
# PBS Pro sets PBS_NCPUS to exactly what the job requested, so the job
# script becomes the single place parallelism is controlled. Falls back to
# a conservative 4 off-scheduler.
#
# There is deliberately NO PBS_NODEFILE branch. Counting its lines is the
# Torque / OpenPBS idiom (one line per core); PBS Pro writes one line per
# CHUNK, so select=1:ncpus=12 yields a one-line file and the count returns
# 1, not 12. Strictly worse than the default below.
#
# Mirrors predict.config.default_n_jobs() so shell and Python agree.
if [[ -n "${PREDICT_N_JOBS:-}" ]]; then
    N_JOBS="$PREDICT_N_JOBS"
elif [[ -n "${PBS_NCPUS:-}" ]]; then
    N_JOBS="$PBS_NCPUS"
elif [[ -n "${PBS_JOBID:-}" && -n "${NCPUS:-}" ]]; then
    # some PBS Pro clusters set NCPUS instead of PBS_NCPUS.
    N_JOBS="$NCPUS"
elif [[ -n "${SLURM_CPUS_ON_NODE:-}" ]]; then
    N_JOBS="$SLURM_CPUS_ON_NODE"
else
    # Deliberately NOT nproc: off the scheduler we are almost certainly on a
    # shared login node, where nproc reports all 64 physical cores rather than
    # the small share this session is entitled to.
    N_JOBS=4
fi

# ── BLAS thread control ───────────────────────────────────────────────
# MUST be exported before any Python starts. Without this, OpenBLAS spawns
# one thread per core INSIDE each joblib worker: 12 workers x 12 BLAS
# threads on 12 cores is heavy oversubscription and is a common reason an
# HPC run finishes slower than a laptop. joblib gives us the parallelism;
# BLAS stays single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

STAGE_FROM=1
STAGE_TO=10
RUN_EXPLORATORY=true
DRY_RUN=false
CONDA_ENV="predict_env"

# ─────────────────────── CLI parsing ───────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)              STAGE_FROM="$2"; shift 2 ;;
        --to)                STAGE_TO="$2"; shift 2 ;;
        --n-jobs)            N_JOBS="$2"; shift 2 ;;
        --skip-exploratory)  RUN_EXPLORATORY=false; shift ;;
        --conda-env)         CONDA_ENV="$2"; shift 2 ;;
        --dry-run)           DRY_RUN=true; shift ;;
        -h|--help)
            grep '^#' "$0" | head -25 | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown flag: $1"; exit 2 ;;
    esac
done

# ─────────────────────── activate conda env ───────────────────────

# Source the conda profile UNCONDITIONALLY. The parent shell may have
# `(predict_env)` in its prompt and CONDA_EXE in the env, but the `conda`
# shell function does NOT survive into a child shell: sub-bash starts
# cold and needs the profile script sourced even if the variables look set.
CONDA_PROFILE=""
for cand in \
    "${PREDICT_CONDA_ROOT:-}" \
    "$HOME/projects/miniconda3" \
    "$HOME/miniconda3" \
    "/opt/miniconda3" \
    "${CONDA_PREFIX:-}/.." \
    "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}"; do
    if [[ -n "$cand" && -f "$cand/etc/profile.d/conda.sh" ]]; then
        CONDA_PROFILE="$cand/etc/profile.d/conda.sh"
        break
    fi
done

if [[ -z "$CONDA_PROFILE" ]]; then
    echo "ERROR: could not find conda.sh profile script. Tried:"
    echo "  \$PREDICT_CONDA_ROOT = ${PREDICT_CONDA_ROOT:-unset}"
    echo "  \$HOME/projects/miniconda3 = $HOME/projects/miniconda3"
    echo "  \$HOME/miniconda3 = $HOME/miniconda3"
    echo "  /opt/miniconda3"
    echo "  \$CONDA_PREFIX/.. = ${CONDA_PREFIX:-unset}/.."
    echo "  \$CONDA_EXE-derived = ${CONDA_EXE:-unset}"
    echo "Edit run_pipeline.sh and add your conda root to the search loop."
    exit 1
fi

source "$CONDA_PROFILE"
conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

# ─────────────────────── logging setup ───────────────────────

LOG_DIR="$PROJECT_ROOT/outputs/_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
MASTER_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

log() {
    local msg="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
    echo "$msg" | tee -a "$MASTER_LOG"
}

run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    local cmd="$3"
    local stage_log="$LOG_DIR/stage${stage_num}_${stage_name}_${TIMESTAMP}.log"

    if [[ "$stage_num" -lt "$STAGE_FROM" || "$stage_num" -gt "$STAGE_TO" ]]; then
        log "SKIP stage $stage_num ($stage_name): outside [--from $STAGE_FROM, --to $STAGE_TO]"
        return 0
    fi

    log "STAGE $stage_num ($stage_name) START"
    log "  cmd : $cmd"
    log "  log : $stage_log"

    if $DRY_RUN; then
        log "  [DRY-RUN] not executing"
        return 0
    fi

    local t_start=$(date +%s)
    set +e
    bash -c "$cmd" >>"$stage_log" 2>&1
    local rc=$?
    set -e
    local t_end=$(date +%s)
    local dur=$((t_end - t_start))

    if [[ $rc -ne 0 ]]; then
        log "STAGE $stage_num ($stage_name) FAILED after ${dur}s (rc=$rc)"
        log "  tail of $stage_log:"
        tail -20 "$stage_log" | tee -a "$MASTER_LOG"
        exit "$rc"
    fi
    log "STAGE $stage_num ($stage_name) OK in ${dur}s"
}

# ─────────────────────── banner ───────────────────────

log "================================================================"
log " PrediCT: pipeline run"
log "================================================================"
log " project root : $PROJECT_ROOT"
log " conda env    : $CONDA_ENV ($(python --version 2>&1))"
log " python path  : $(which python)"
log " stages       : $STAGE_FROM..$STAGE_TO"
log " --n-jobs     : $N_JOBS"
log " exploratory  : $RUN_EXPLORATORY"
log " dry run      : $DRY_RUN"
log " master log   : $MASTER_LOG"

# ─────────────────────── stages 1..3 ───────────────────────

run_stage 1 "io_discovery" \
    "python scripts/01_discover.py"

run_stage 2 "preprocess" \
    "python scripts/02_preprocess.py"

run_stage 3 "features" \
    "python scripts/03_features.py"

# Audits of the published stage 3 outputs (row count vs manifest, schema,
# NaN policy, per-vessel totals, native per-vessel masks union = whole mask).
# Exits 1 on any failed audit, which stops the pipeline.
run_stage 3 "features_sanity" \
    "python scripts/03b_cohort_sanity.py"

# ─────────────────────── stage 4 (stability gates) ───────────────────────

# D014, kept as the supplementary comparison gate.
run_stage 4 "perturbations_d014" \
    "python scripts/04_perturbations.py"

run_stage 4 "icc_gate_d014" \
    "python scripts/05_icc_gate.py"

# D033, the primary gate (stability.primary_gate: realistic).
run_stage 4 "calibrate_d033" \
    "python scripts/calibrate_realistic.py"

run_stage 4 "perturbations_d033" \
    "python scripts/04b_realistic_perturbations.py --arms all"

run_stage 4 "icc_gate_d033" \
    "python scripts/05b_icc_gate_realistic.py"

run_stage 4 "robustness_tiers_d034" \
    "python scripts/05c_robustness_tiers.py"

# ─────────────────────── stage 5 (reduce, x3 cohorts) ───────────────────────

run_stage 5 "reduce_full" \
    "python scripts/06_reduce.py --n-jobs $N_JOBS"

run_stage 5 "reduce_Qr36d_2" \
    "python scripts/06_reduce.py --kernel-filter 'Qr36d/2' --n-jobs $N_JOBS"

run_stage 5 "reduce_I30f_3" \
    "python scripts/06_reduce.py --kernel-filter 'I30f/3' --n-jobs $N_JOBS"

# ─────────────────────── stage 6 (discover, x3 cohorts) ───────────────────────

run_stage 6 "discover_full" \
    "python scripts/07_discover.py --cohort-dir outputs/06_reduce/ --n-jobs $N_JOBS"

run_stage 6 "discover_Qr36d_2" \
    "python scripts/07_discover.py --cohort-dir outputs/06_reduce/stratified_Qr36d_2/ --n-jobs $N_JOBS"

run_stage 6 "discover_I30f_3" \
    "python scripts/07_discover.py --cohort-dir outputs/06_reduce/stratified_I30f_3/ --n-jobs $N_JOBS"

# ─────────────────────── stage 7 (analyse) ───────────────────────

run_stage 7 "analyse" \
    "python scripts/08_analyse.py"

run_stage 7 "analyse_report" \
    "python scripts/08d_stage7_results.py | tee outputs/07_analyse/stage7_results_report.txt"

run_stage 7 "analyse_verify" \
    "python scripts/08e_verify_focal_diffuse.py | tee outputs/07_analyse/focal_diffuse_verification.txt"

# ─────────────────────── stage 8 (validate) ───────────────────────

run_stage 8 "validate" \
    "python scripts/09_validate.py"

# ─────────────────────── stage 9 (lesion chain, exploratory) ───────────────────────

if $RUN_EXPLORATORY; then
    # The chain writes into outputs/exploratory.partial; `finish` verifies it
    # and publishes it as a whole with COMPLETE_stage9.json (D043). A failing
    # step stops the pipeline before `finish`, so the previous complete
    # result stays published.
    export PREDICT_STAGE9_ROOT=exploratory.partial
    run_stage 9 "stage9_begin" \
        "python scripts/09_stage9.py begin"
    run_stage 9 "lesion_morph_run" \
        "python experiments/lesion_morphology/run.py --n-jobs $N_JOBS"
    run_stage 9 "lesion_morph_analyse" \
        "python experiments/lesion_morphology/analyse.py --n-jobs $N_JOBS"
    run_stage 9 "lesion_morph_finalise" \
        "python experiments/lesion_morphology/finalise.py --n-jobs $N_JOBS"
    run_stage 9 "lesion_k_sensitivity" \
        "python experiments/lesion_morphology/k_sensitivity.py --n-jobs $N_JOBS"
    run_stage 9 "lesion_c8_other" \
        "python experiments/lesion_morphology/c8_other_lesions.py"
    run_stage 9 "lesion_c8_matched" \
        "python experiments/lesion_morphology/c8_burden_matched.py"
    run_stage 9 "lad_run" \
        "python experiments/lad_phenotype/run.py"
    run_stage 9 "lad_run_qr" \
        "python experiments/lad_phenotype/run.py --stratum 'Qr36d/2'"
    run_stage 9 "lad_run_i30" \
        "python experiments/lad_phenotype/run.py --stratum 'I30f/3'"
    run_stage 9 "lad_matched" \
        "python experiments/lad_phenotype/matched.py"
    run_stage 9 "lad_matched_qr" \
        "python experiments/lad_phenotype/matched.py --stratum 'Qr36d/2'"
    run_stage 9 "lad_matched_i30" \
        "python experiments/lad_phenotype/matched.py --stratum 'I30f/3'"
    run_stage 9 "lad_finalise" \
        "python experiments/lad_phenotype/finalise.py"
    run_stage 9 "lm_isolated_run" \
        "python experiments/lm_isolated_low_burden/run.py"
    run_stage 9 "lm_isolated_finalise" \
        "python experiments/lm_isolated_low_burden/finalise.py"
    run_stage 9 "lesion_stability_d033" \
        "python experiments/lesion_morphology/lesion_stability_d033.py --n-jobs $N_JOBS"
    run_stage 9 "stage9_finish" \
        "python scripts/09_stage9.py finish"
    unset PREDICT_STAGE9_ROOT
fi

# ─────────────────────── stage 10 (figures) ───────────────────────

run_stage 10 "figures" \
    "python scripts/10_figures.py"

# ─────────────────────── stage 11 (self-checks, on request) ───────────────────────

run_stage 11 "kernel_leak" \
    "python scripts/diagnose_kernel_leak.py production --out outputs/checks/kernel_leak"

run_stage 11 "kernel_chi2" \
    "python scripts/diagnose_kernel_chi2.py production --out outputs/checks/kernel_chi2"

run_stage 11 "detectability_d0361" \
    "python scripts/probe_detectability_floor.py --root outputs --amendment"

# ─────────────────────── done ───────────────────────

log "================================================================"
log " PIPELINE COMPLETE"
log "================================================================"
log " all requested stages OK; outputs under $PROJECT_ROOT/outputs/"
log " master log: $MASTER_LOG"
log ""
log " quick spot-check commands:"
log "   ls outputs/01_manifest/manifest.csv"
log "   ls outputs/03_features/features.csv"
log "   cat outputs/05b_icc_realistic/summary.txt"
log "   ls outputs/06_reduce/{representative_features,cohort_metadata,prepared_matrix}.csv"
log "   ls outputs/07_analyse/{phenotype_paper_table,directional_verdict.json,burden_orthogonality}.csv"
log "   cat outputs/07_analyse/directional_verdict.json"
log "   ls outputs/08_validate/{external_holdout_report,leave_k_out_ari,cross_cohort_ari_consolidated}.csv"
log "   cat outputs/exploratory/lesion_morphology/k_sensitivity/summary.txt"
