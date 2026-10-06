# D025 Stage 7 directional hypothesis test

**Date**: 2026-06-06
**Stage**: analyse (stage 7)
**Status**: Active
**Module**: `src/predict/analyse/hypotheses.py`

## Decision

Six pre-specified, pre-registered, one-sided Mann-Whitney U tests on the focal-vs-diffuse comparison for the spatial-only k=2 phenotype. The 6 hypotheses are locked from the design discussion (informed by Hoori 2024 / Lin 2022) before any stage 7 numerical result was observed:

| # | Feature | Predicted direction (focal vs diffuse) | Clinical interpretation |
|---|---|---|---|
| 1 | `lesion_count_lad` | focal > diffuse | focal disease concentrates in the LAD |
| 2 | `n_calcified_arteries` | focal < diffuse | focal disease is single-vessel |
| 3 | `dist_from_top_max` | focal < diffuse | focal disease is proximal (closer to ostium) |
| 4 | `gini_lesion_volume` | focal > diffuse | focal disease has concentrated lesion-volume distribution |
| 5 | `vessel_burden_gini` | focal > diffuse | focal disease concentrates burden in one vessel |
| 6 | `first_to_last_dist_lad` | focal < diffuse | focal disease is compact along the LAD axis |

For each hypothesis we compute:

| Column | Content |
|---|---|
| `feature` | the canonical feature name |
| `predicted_direction` | "focal>diffuse" or "focal<diffuse" |
| `focal_median`, `diffuse_median` | raw medians per cluster |
| `observed_sign` | sign(focal_median - diffuse_median): -1, 0, or +1 |
| `direction_match` | True iff observed_sign matches predicted direction |
| `mannwhitney_u_pval_one_sided` | one-sided Mann-Whitney p, ALTERNATIVE matches predicted direction |
| `cliffs_delta` | rank-based effect size, same as D023 |
| `fdr_bh_pval` | BH-adjusted p across the 6 directional tests |
| `confirmed` | True iff direction_match AND fdr_bh_pval < 0.05 |

Two-tier verdict:

```
PRIMARY pass (full cohort N=420):
  count(confirmed) >= 4 of 6 hypotheses

SECONDARY pass (kernel-stratified replication):
  In Qr36d/2 stratum (N=220): count(direction_match) >= 4
  AND
  In I30f/3 stratum (N=200): count(direction_match) >= 4
  (significance NOT required; direction only)

OVERALL VERDICT:
  primary_pass + secondary_pass  -> "robust"
  primary_pass + secondary_fail  -> "kernel-confounded"
  not primary_pass               -> "refuted"
```

## Rationale

### Why one-sided

A directional hypothesis is pre-specified BEFORE the test. The alternative is one-sided by construction; using two-sided would be doubling the p-value for a result we have committed to interpreting in only one direction. One-sided Mann-Whitney is standard for pre-registered directional predictions.

### Why FDR-BH across the 6, not the 41

The 6 directional hypotheses are a closed, pre-specified family with their own multiple-testing concern. FDR-BH on the 6 controls the directional-family false discovery rate at alpha = 0.05. The wider 41-feature signature_features.csv has its own FDR-BH bundle (per D023).

### Why >= 4 of 6, not all 6

- Pre-registering an "all 6 must confirm" criterion is too brittle: a single feature that fails for any reason (poor stage-3 implementation, kernel-specific quirk, sentinel pattern) refutes the whole finding.
- >= 4 of 6 (66.7%) is a substantial majority that allows for 1-2 expected "noise" hypotheses while still requiring the bulk of the directional model to hold.
- This threshold is pre-registered along with the hypotheses themselves.

### Why secondary criterion is direction-only

- The kernel strata are small enough (N=220, N=200) that requiring significance in BOTH would over-discount real signal.
- The point of the secondary check is to detect "spurious primary pass driven by kernel imbalance": if the full-cohort primary passes only because the kernel-mix happens to align with the predicted directions but the strata individually disagree on direction, the result is fragile.
- Direction-only is the correct level of strictness for this consistency check.

### Why this 6-hypothesis list

- They are independent enough that one feature failing does not necessarily drag the others down (e.g., `lesion_count_lad` (count) and `dist_from_top_max` (distance) measure different things even though both describe the LAD focal pattern).
- They cover the four geometric dimensions of "focal" disease per Hoori 2024 / Lin 2022: vessel concentration (#2, #5), proximal-vs-distal (#3), volume concentration (#4), per-vessel compactness (#1, #6).
- The list was finalised BEFORE any stage 7 numerical result was observed and BEFORE any of these features were ranked by Cliff's delta.

## Alternatives considered

- **All 6 must confirm**: rejected as too brittle.
- **Bayesian directional inference (posterior P(direction | data))**: rejected; we want a frequentist criterion the radiomics literature recognises.
- **Two-sided Mann-Whitney with direction reported separately**: rejected; pre-registered directional tests should be one-sided.
- **Skip the secondary stratified criterion**: rejected; full-cohort pass alone is vulnerable to kernel-imbalance confounding.

## Verified by

- `tests/analyse/test_hypotheses.py` covering: one-sided Mann-Whitney sign matches predicted; FDR-BH adjustment on 6 known p-values; primary verdict on 4/6 vs 3/6; secondary verdict triggers "kernel-confounded" when one stratum agrees and the other disagrees; integration test on a synthetic cohort with planted focal-vs-diffuse structure.
