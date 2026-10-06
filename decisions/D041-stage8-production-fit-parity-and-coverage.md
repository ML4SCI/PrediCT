# D041 Stage 8 integrity: production fit, parity, complete coverage, all-or-nothing publication

**Date**: 2026-10-05
**Stage**: validate (stage 8)
**Status**: Active (amends D029 and D030; their questions, holdout set, fold design, threshold rule and descriptive scope are unchanged)
**Module**: `src/predict/validate/external_holdout.py`, `src/predict/validate/leave_k_out.py`, `src/predict/validate/stage8.py`, `scripts/09_validate.py`, `src/predict/reduce/prepare_matrix.py`, `src/predict/discover/cluster_discovery.py`

## Decision

1. **Production fit.** D029 and D030 refit the spatial pipeline with the production settings: stage 5's `production_prep_kwargs(cfg)` (shared with 06_reduce), the configured PCA variance, and stage 6's `random_state` with stage 6's own GMM function `fit_gmm` (5 initialisations, 200 iterations). D030.1's literal "GMM random_state=0, n_init=10" is replaced: D030 validates the production discovery pipeline, and those values were not the production ones.
2. **Exact replay.** Fitted transforms are kept at full precision (Yeo-Johnson lambda, z-score mean and standard deviation), and a rank-fallback column keeps its training values, so a new value receives the training rank mapping (a value equal to a training value gets that value's rank). Every fit checks that replaying its training rows through the held-out transform reproduces the fitted matrix (tolerance 1e-9) and the GMM labels.
3. **Parity with the published run (D029).** The refit on the published cohort must reproduce stage 5's prepared spatial matrix (1e-9) and stage 6's spatial labels exactly, in the published patient order, before any holdout patient is projected.
4. **Exact cohort and coverage.** D029 and D030 use exactly stage 5's published cohort (prepared_matrix.csv order). D030 requires a reference label for every cohort patient and a row for every planned fold. Reference focal/diffuse counts use the published mapping (D023 on the full-cohort reference labels), not the fold model's label numbers. ARI is unchanged (raw labels, D030.5).
5. **Complete holdout.** The holdout set comes from stage 1's manifest (no fixed fallback list). Every expected patient must be preprocessed and featurised; any processing failure is written to `holdout_failures.csv` and stops the stage. D002 round-trip failures stay descriptive (D029.3).
6. **Publication.** Inputs are verified first (stage 7 record, which covers stages 5-6 and features.csv; gate files equal to stage 5's record). Outputs go to `08_validate.partial/` and are published with `COMPLETE_stage8.json` only when every deliverable completed; any failure exits non-zero and publishes nothing. Skipped deliverables or non-production CV settings write to `outputs/08_validate_smoke/` with `PARTIAL_SMOKE_DO_NOT_USE`.

## Rationale

An aborted holdout exited 0; failed patients were skipped and the holdout could report "ok" with fewer patients; outputs mixed across runs. Rank-fallback columns were left raw and z-scored with rank-scale statistics (658.77 instead of 3.28 in a synthetic example); parameters were rebuilt from rounded logs. The "frozen" model used other GMM settings without a parity requirement (ARI 1.0 on the current data, but nothing enforced it). D030 ran on 437 patients against a 436-patient reference, silently dropping the unmatched patient, passed a synthetic run with 25% of reference labels missing, and could report reversed reference class counts after a GMM label swap.

## Interpretation (unchanged, stated for the record)

The GE scans assess technical transfer within this dataset (N=4, descriptive). Cross-validation assesses reproducibility of the partition. Neither establishes clinical validity or distinct biological subtypes.

## Verified by

- `tests/validate/test_external_holdout.py`: rank fallback replayed with the training mapping; the fit refuses a wrong transform; production parity passes and detects a label flip, a 1e-6 transform shift and a reordered cohort; the frozen GMM is stage 6's.
- `tests/validate/test_leave_k_out.py`: incomplete reference coverage refused; reference counts unchanged by swapped label numbers.
- `tests/validate/test_orchestrator.py`: manifest required; a holdout processing failure stops the stage and is recorded; a complete run publishes with a record, a failing run keeps the previous result, a reduced run goes to the smoke folder.
