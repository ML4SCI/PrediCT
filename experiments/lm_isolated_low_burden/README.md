# Exploratory: isolated left-main calcification at low total burden

**Status**: exploratory, pre-registered (`plan.md`), run as part of stage 9
(`run_pipeline.sh`). Produces no file consumed by stages 1 to 8 and does not
modify any decision. Results: [findings.md](findings.md).

## Question

Within the lowest Agatston tertile, are there patients whose feature
profile places them away from the low-burden core of the patient PCA, and
is that group enriched for left-main (LM) calcification? If so, are their
LM lesions soft or dense, and are they distinct from the LAD-signature
lesion clusters of `../lad_phenotype/`?

## Criteria (fixed in `plan.md`)

| Step | Rule |
|---|---|
| P1 | displaced = low tertile AND (PC1 > 0 OR PC2 > 2.5) |
| P2 | displaced LM rate >= 0.5 and one-sided Fisher p < 0.001 against the other low-tertile patients |
| P3 | displaced LM rate >= 0.5 within each kernel stratum |
| P4 | density profile of the displaced patients' LM lesions (descriptive) |
| P5 | < 20% of those LM lesions in this run's LAD-signature clusters; a missing or empty signature makes P5 not evaluable (D043) |

## Scripts and outputs

```bash
python experiments/lm_isolated_low_burden/run.py       # P1-P5
python experiments/lm_isolated_low_burden/finalise.py  # report.txt
pytest experiments --import-mode=importlib             # tests
```

Outputs under `outputs/exploratory/lm_isolated_low_burden/`:

| File | Content |
|---|---|
| `displaced_patients.csv` | the displaced patients with PC scores and vessel burden |
| `full_cohort_displacement.csv` | every cohort patient with PC scores and displaced flag |
| `fisher_test.json` | P2 table, Fisher result, verdict |
| `cross_stratum.json` | P3 per-stratum rates, Wilson intervals, verdicts |
| `density_profile.json` | P4 density tiers of the LM lesions |
| `cluster_overlap.json` | P5 per-lesion cluster ids and overlap |
| `summary.json` | all verdicts |
| `run_header.json` | library versions, input hashes, plan hash, thresholds |
| `report.txt` | human-readable report |
