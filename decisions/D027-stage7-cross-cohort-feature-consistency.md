# D027 Stage 7 cross-cohort feature-level consistency criterion

**Date**: 2026-06-06
**Stage**: analyse (stage 7)
**Status**: Active
**Module**: `src/predict/analyse/cross_cohort.py`

## Decision

A feature qualifies as a **robust cross-cohort discriminator** for a given (partition x cluster) cell if and only if ALL three of these criteria hold simultaneously across the full / Qr36d/2 / I30f/3 cohort outputs:

```
Rule 1: direction consistent
  sign(cliffs_delta) is the same in all 3 cohorts
  (a feature that is "higher in focal" in two cohorts and "higher in diffuse"
   in the third is NOT a robust discriminator; it is cohort-specific)

Rule 2: significance in at least 2 of 3
  fdr_bh_pval < 0.05 in at least 2 of the 3 cohorts
  (allows for the smallest stratum to lose significance from power loss,
   but requires the bulk of evidence to be significant)

Rule 3: minimum effect size in all 3
  abs(cliffs_delta) >= 0.20 in all 3 cohorts
  (requires the effect to be at least "small-to-medium" magnitude
   in every cohort, not just on average)
```

Per-feature output `cross_cohort_feature_consistency.csv` columns:

| Column | Content |
|---|---|
| `feature`, `partition`, `cluster` | identifies the comparison |
| `sign_full`, `sign_qr36d_2`, `sign_i30f_3` | sign(cliffs_delta) per cohort |
| `delta_full`, `delta_qr36d_2`, `delta_i30f_3` | abs(cliffs_delta) per cohort |
| `fdr_pval_full`, `fdr_pval_qr36d_2`, `fdr_pval_i30f_3` | FDR-adjusted Mann-Whitney p per cohort |
| `rule1_direction_consistent` | True / False |
| `rule2_significance_in_2of3` | True / False |
| `rule3_min_effect_size` | True / False |
| `robust_discriminator` | True iff all three rules pass |

Separately, the partition-level **ARI on shared pids** check is preserved and complements (but does not replace) the feature-level criterion:

```
For each partition (spatial_k2, burden_k3):
  Take the intersection of pids in {full, Qr36d/2} and {full, I30f/3}
  Compute ARI between the cluster labels assigned to those shared patients
  PASS: ARI >= 0.80 for both stratified-vs-full pairs
```

Output `cross_cohort_ari.csv` records the partition-level ARI for each
(stratified cohort) pair.

## Rationale

### Why three rules instead of one

A single composite criterion (e.g., FDR-adjusted-p < 0.05 in all 3) would either be too lenient or too strict depending on stratum size:

- Too lenient: a feature with delta = 0.05 in two cohorts and delta = 0.40 in one would pass FDR significance in 3 of 3 cohorts in a 420-patient setting; but the effect is dominated by one cohort. The "in all 3" version of rule 3 (effect size threshold per cohort) blocks this.
- Too strict: requiring FDR-adjusted significance in 3 of 3 over-rejects features that fail in the 200-patient I30f/3 stratum purely from power loss. Rule 2 ("2 of 3") allows this.

The three rules together are a stable, replication-grade discriminator definition: the direction is consistent (rule 1), the bulk of evidence is significant (rule 2), and the effect is at least small-to-medium in every cohort (rule 3).

### Why effect-size rule has the strict "in all 3" form

- Rule 1 (direction) and Rule 2 (significance) together still admit a feature with sign-consistent but trivial effect in one cohort. Rule 3 is the safety net.
- |delta| >= 0.20 is the Romano 2006 "small-to-medium" floor, the same threshold used in D023's `is_robust_discriminator` column.

### Why the partition-level ARI is separate

- ARI on shared pids tests whether the SAME PATIENT lands in the SAME PHENOTYPE across cohort runs. This is partition identity, which is about cluster definitions agreeing, not feature discriminator agreeing.
- A feature can be a robust discriminator under D027 even if the partition ARI is low (different patients map differently, but the features that distinguish whatever-cluster-A is from whatever-cluster-B is are stable). Conversely, partition ARI can be high while individual features differ.
- Both checks belong in the paper as separate evidence pillars.

### Why ARI threshold is 0.80

- ARI ranges [-1, 1]; 0 is chance agreement; 1.0 is perfect identity. 0.80 is a strong replication threshold (Hubert & Arabie 1985 / Steinley 2004 convention).
- 0.75 would be too lenient given the stratified cohorts are subsets of the full cohort (so a strong replication is expected).
- 0.90 would be too strict given that some patients near the GMM decision boundary will reassign across cohort runs.

## Alternatives considered

- **Use only ARI (partition-level), no feature-level criterion**: rejected; ARI tells you partition identity replicates, NOT that any given feature is a reliable discriminator. The paper needs both.
- **Use only feature-level (3 rules), no ARI**: rejected; without partition ARI, we can't claim "cluster A in the full cohort IS cluster A in the strata", only that "the cluster-A signature has the same features".
- **2-rule criterion (drop rule 3)**: rejected; rule 3 is the safety against effect-size dilution.
- **Allow rule 1 to be "in 2 of 3" instead of "in all 3"**: rejected; cross-cohort direction inconsistency is exactly what we want to flag, never accept.

## Verified by

- `tests/analyse/test_cross_cohort.py` covering: each rule individually on synthetic; combined criterion on a feature designed to pass / fail each rule; partition ARI matches `discover.validity.ari_on_shared_pids`; empty intersection of pids raises.
