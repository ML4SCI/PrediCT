# D013 ICC formulation and threshold

**Date**: 2026-06-03
**Stage**: stability
**Status**: Active
**Module**: `src/predict/stability/icc.py`

## Decision

Use **ICC(3,1), absolute agreement, two-way mixed-effects, single-measurement** as the reliability statistic for every feature that goes through the perturbation gate.

Pass threshold: **ICC >= 0.75**. Features below 0.75 are dropped from the analysis-ready feature set written by stage 5.

The formula (Shrout and Fleiss 1979 / McGraw and Wong 1996):

```
ICC(3,1) = (MSR - MSE) / (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE))
```

where `n` is the number of patients, `k` is the number of perturbations (14 per D014), `MSR` is the between-patients mean square, `MSC` is the between-perturbations mean square, and `MSE` is the residual mean square. Computed per feature on the 422-by-14 reliability matrix (D015).

Output schema (`outputs/05_icc/icc_report.csv`):

```
feature                          icc      icc_source                  passes_gate
lesion_count_lad                 1.000    invariant_by_construction   True
agatston_lad                     0.987    empirical                   True
original_glcm_Contrast           0.823    empirical                   True
original_glrlm_GrayLevelVariance 0.612    empirical                   False
```

## Rationale

- **ICC(3,1), not ICC(2,1) or ICC(1,1)**. The 14 perturbations are a fixed, deterministic set (not random draws), so the "raters" (perturbations) are a fixed effect. This is the textbook condition for ICC(3,k). We compute single-measurement (k=1, not mean of k) because downstream code consumes one feature value per patient, not an average across perturbations.
- **Absolute agreement, not consistency**. Consistency ICC ignores systematic offsets between raters; absolute agreement penalises them. A rotation that uniformly biases a feature by a fixed amount is still a reliability concern (it means the feature depends on patient pose), so we penalise it.
- **Threshold 0.75**. Koo and Li 2016 classify ICC < 0.50 as poor, 0.50 to 0.75 as moderate, 0.75 to 0.90 as good, > 0.90 as excellent. 0.75 is the standard floor in radiomics reproducibility studies (Zwanenburg 2020 IBSI-2, Lin 2022, Kolossvary 2025).
- **Single threshold for all features**. A per-feature-family threshold (e.g., 0.85 for texture, 0.75 for first-order) would be defensible but introduces tunable knobs. We avoid that for a publication-grade pipeline; one number, one rule.

## Alternatives considered

- **ICC(2,1)**. Treats perturbations as random raters drawn from a population. Wrong here: our 14 perturbations are not a random sample, they are a defined design matrix. Rejected.
- **ICC(1,1)**. One-way model, ignores rater (perturbation) identity. Wastes design information and produces lower ICCs by lumping perturbation variance into residual. Rejected.
- **CCC (Lin's concordance correlation coefficient)**. Pairwise only, requires choosing a reference perturbation, awkward for k=14. Rejected.
- **Threshold 0.85 (stricter)**. Halves the surviving PyRadiomics set on typical NCCT calcium studies (Mackin 2015). Would risk losing biologically meaningful but moderately noisy texture features. A 0.85 count is reported as a sensitivity table (`scripts/05c_threshold_sensitivity.py`).
- **Threshold 0.50 (more permissive)**. Lets in features that are only moderately stable. Defeats the purpose of the gate.

## Verified by

- `tests/stability/test_icc.py`: unit tests for the closed-form ICC(3,1) on known toy matrices (constant column => ICC=1; orthogonal noise => ICC near 0; one strong fixed effect across raters => penalised under absolute agreement).
- `scripts/05_icc_gate.py`: produces `icc_report.csv` with the threshold applied; the row count of `gated_features.csv` matches the count of features at or above 0.75.
