# Stage 8: validate

## What this stage does

`scripts/09_validate.py` produces three deliverables in `outputs/08_validate/`:

- **D029 external GE holdout** (`external_holdout_report.csv`,
  `xml_roundtrip_holdout.csv`, `holdout_features.csv`): the GE patients
  excluded by D004 are preprocessed and featurised with the stage 2-3 code,
  then projected onto the production spatial-only PCA and assigned a
  spatial k=2 phenotype by the production GMM. Descriptive only (N=4).
- **D030 leave-k-out cross-validation** (`leave_k_out_ari.csv`): 10-fold
  kernel-stratified folds; each fold refits the spatial pipeline on its
  training rows and predicts the held-out rows; ARI against the published
  stage 6 labels; pass threshold from a K=10% disagreement simulation.
- **D031** (`cross_cohort_ari_consolidated.csv`): re-export of stage 7's
  cross-cohort ARI with pass columns.

## Integrity rules (D041)

- **Inputs first.** `predict.validate.stage8.stage8_inputs` requires a
  complete, current stage 7 record (which covers the stage 6 records, the
  stage 5 files and features.csv) and gate files equal to the hashes in
  stage 5's record.
- **Production fit.** D029 and D030 use `production_settings(cfg)`:
  `production_prep_kwargs(cfg)` (the function 06_reduce uses), the
  configured PCA variance, stage 6's random_state and `fit_gmm` (stage 6's
  GMM). Cohort = exactly stage 5's published cohort.
- **Exact replay.** Full-precision Yeo-Johnson lambdas and z-score
  parameters; rank-fallback columns map new values through the training
  values. Every fit replays its training rows and must reproduce the fitted
  matrix (1e-9) and labels.
- **Parity (D029).** The refit on the published cohort must reproduce stage
  5's prepared spatial matrix and stage 6's spatial labels before the
  holdout is projected.
- **Coverage.** Every holdout patient from stage 1's manifest must process;
  failures go to `holdout_failures.csv` and stop the stage (D002 round-trip
  failures stay descriptive). D030 needs a reference label for every cohort
  patient and every planned fold; reference class counts use the published
  focal/diffuse mapping.
- **Publication.** `08_validate.partial/` is published with
  `COMPLETE_stage8.json` only after a complete run; failures exit non-zero
  and publish nothing. Reduced runs (`--skip-*`, non-default `--n-splits`,
  `--n-simulations`, `--seed`) write to `outputs/08_validate_smoke/`.

## Results on the analysis cohort

- **D029 GE holdout.** 4 of 4 GE scans processed. Production parity before
  projection: 436 patients, largest transform difference 8.9e-16, 0 label
  mismatches. Assignments follow burden: the two low-burden scans (Agatston
  45 and 43) are focal, the two high-burden scans (970 and 1145) diffuse.
  Two holdout scans fail the XML Max round-trip (descriptive, D029).
- **D030 leave-k-out.** Median ARI .820 against a pass threshold of .660:
  PASS by the D030 rule, but 3 of 10 folds fall below their threshold and
  the IQR is .32. The split reproduces on most refits, not all.
- **D031.** Cross-cohort ARI as in stage 7 (2 of 4 comparisons pass at
  0.80).

## Interpretation

The GE scans test technical transfer within COCA, not generalisation to
other populations. Cross-validation tests reproducibility of the spatial
partition, not clinical validity or the existence of distinct subtypes.

## How to run

```bash
python scripts/09_validate.py                 # production, publishes
python scripts/09_validate.py --skip-holdout  # smoke folder only
```

## Decisions

- D029 external GE holdout; D030 leave-k-out CV; D031 ARI consolidation;
  D041 production fit, parity, coverage and publication.
