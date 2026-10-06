# D026 Stage 7 monotonicity test and burden-axis classification

**Date**: 2026-06-06
**Stage**: analyse (stage 7)
**Status**: Active
**Module**: `src/predict/analyse/monotonicity.py`

## Decision

For each of the 28 robust kernel-independent features (the cross-cohort intersection from stage 5 / 6), compute Spearman rho and Kendall tau against `agatston_total` (RAW, not log-transformed). Classify each feature into one of four mechanistic groups based on the relationship strength and the feature's stage-5 block:

| Group | Classification rule | Interpretation |
|---|---|---|
| `burden_tracking` | `abs(spearman_rho) >= 0.5` | Feature is essentially a re-parameterisation of total calcium burden |
| `structure_tracking` | `abs(spearman_rho) < 0.3` AND block in {`hu_statistics`, `texture`, `shape`} | Feature describes calcium morphology / intensity, weakly tied to burden |
| `spatial_tracking` | `abs(spearman_rho) < 0.3` AND block == `spatial` | Feature describes lesion geometry, independent of burden |
| `mixed` | not any of the above | Intermediate; needs case-by-case interpretation |

Both Spearman rho and Kendall tau are reported per feature per cohort with p-values; classification is decided by Spearman (more familiar in radiomics literature; Kendall tau is reported for sensitivity).

Output `monotonicity_classification.csv` columns:

| Column | Content |
|---|---|
| `feature` | the 28-rep canonical name |
| `block` | the D022 multi-block partition |
| `cohort` | full / Qr36d/2 / I30f/3 |
| `spearman_rho` | rank correlation with agatston_total |
| `spearman_p` | scipy Spearman p-value |
| `kendall_tau` | Kendall tau-b (handles ties) |
| `kendall_p` | scipy Kendall p-value |
| `classification` | one of burden_tracking / structure_tracking / spatial_tracking / mixed |

Plus a summary table per cohort showing the count of features in each class.

The classification informs the paper's description of Finding 1 (burden dominance): the robust features divide cleanly into a small burden axis, a small spatial-topology axis, and a moderate structure-character axis.

## Rationale

### Why Spearman primary, Kendall sensitivity

- Spearman rho is the standard radiomics rank correlation; readers expect it.
- Kendall tau-b (concordant pairs - discordant pairs, normalised; handles ties correctly) is more robust to outliers and more interpretable in small samples but less familiar.
- We pick Spearman for the classification rule and report Kendall as a sensitivity column. If Spearman and Kendall disagree on classification for any feature, flag it manually for the paper.

### Why these thresholds (|rho| >= 0.5 for burden, < 0.3 for non-burden)

- |rho| >= 0.5 is the conventional "moderate-to-strong" threshold in the radiomics literature (Mukaka 2012). A feature that correlates with burden at rho >= 0.5 is empirically a burden surrogate.
- |rho| < 0.3 is the conventional "weak / negligible" boundary. Below this we can say "not burden-driven".
- The 0.3-0.5 mixed band is where biological interpretation is ambiguous; we mark these features explicitly rather than forcing a classification.

### Why classification uses block AND |rho|

- A spatial feature with |rho| = 0.4 (in the mixed band) is biologically a spatial feature; it should not be re-classified as burden-tracking just because of a moderate correlation. The block tag carries the prior structural information.
- For weak-rho features, the block tag is the tiebreaker. For strong-rho features (>= 0.5), the burden association is empirically dominant regardless of the block tag.

### Why raw (not log) agatston

- The classification is about whether a feature TRACKS burden. Log-transforming agatston shifts the correlation but does not change its sign or its rank ordering. Spearman rho is invariant under monotone transformation; Spearman(rho, agatston) == Spearman(rho, log(agatston + 1)).
- We avoid log to match the rest of stage 7 (raw values everywhere for clinical interpretability).

## Alternatives considered

- **Pearson correlation**: rejected; Pearson is sensitive to outliers and Agatston is heavy-tailed.
- **Linear regression with R^2**: rejected; assumes linear structure, and Agatston-vs-features is monotonic but rarely linear.
- **Cluster the 28 features by their Spearman-with-everything-else profile and assign block labels post-hoc**: rejected; the D022 block partition is prospective and binding (the entire point of multi-block clustering was to use prospective biological groups, not to discover them post-hoc).

## Verified by

- `tests/analyse/test_monotonicity.py` covering: Spearman matches scipy reference; Kendall matches scipy reference; classification rule on synthetic with known rho per group; mixed band correctly catches |rho| = 0.4; summary count table sums to 28.
