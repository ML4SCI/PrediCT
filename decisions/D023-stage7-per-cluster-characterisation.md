# D023 Stage 7 per-cluster characterisation method

**Date**: 2026-06-06
**Stage**: analyse (stage 7)
**Status**: Active
**Module**: `src/predict/analyse/profiles.py`

## Decision

Per-cluster characterisation operates on RAW (not z-scored) feature values for clinical interpretability, and uses **non-parametric** statistics throughout because the cohort has heavy-tailed Agatston scores, sparse-zero density tiers, and skewed PyRadiomics texture distributions where Gaussian-assumption tests (Cohen's d, Welch's t) are misleading.

For each (cohort x partition x cluster x feature) tuple we compute:

| Statistic | Formula / library |
|---|---|
| `n` | cluster size |
| `n_nonzero` | count of nonzero raw values |
| `median` | numpy.median on RAW values (after pid alignment) |
| `iqr_lower`, `iqr_upper` | 25th and 75th percentiles |
| `cliffs_delta` | per-feature effect size: focal-vs-(everyone-else); range [-1, 1] |
| `mannwhitney_u_pval` | two-sided Mann-Whitney U test, focal-vs-(everyone-else) |
| `fdr_bh_pval` | Benjamini-Hochberg-adjusted Mann-Whitney p across the 41 comparisons (28 robust features + 13 spatial inputs) within the same (cohort x partition) bundle |
| `is_robust_discriminator` | True iff fdr_bh_pval < 0.05 AND abs(cliffs_delta) >= 0.20 |

Plus two cohort-level gates that fail-loud the pipeline:

- **Biological sanity check** (per cohort, spatial partition): median `max_hu_global` in the focal cluster must be >= 130 HU (the IBSI / Agatston calcium definition). This is an ABSOLUTE FLOOR, not a relative ratio. A relative ratio rule (focal >= 0.9 * diffuse) fires false-positively on the real cohort, where focal clusters have legitimately lower peak HU (387) than diffuse (744) by ~50% as a biological phenomenon (focal disease is earlier-stage / softer plaque). The 130-HU absolute floor catches the actual failure mode we care about (soft-tissue voxels in the focal cluster -> max_hu near 0 or negative) without firing on real biology. Additionally, if focal_median / diffuse_median < 0.5, log a WARNING (do not raise) so unexpected biology is visible in the audit log.
- **Label balance check** (per cohort, every partition): minority class size must be > 15% of the cohort. If a partition ends up >= 85/15 imbalanced, the smaller cluster's median + IQR statistics are too noisy to publish.

Both checks raise at the orchestrator entry, before any output file is written.

## Rationale

### Cliff's delta over Cohen's d

- Cliff's delta is the proportion of (focal, diffuse) pairs where focal > diffuse, minus the proportion where focal < diffuse. It is exactly zero when the two distributions are identical (or perfectly overlap), and 1.0 (-1.0) only when every focal value exceeds (is below) every diffuse value. It is rank-based, dimensionless, and robust to outliers and skew.
- Cohen's d on z-scored data is the radiomics-publication convention but assumes Gaussian inputs. On 93%-sparse LM density-tier bins, Cohen's d explodes to physically meaningless values, and on heavy-tailed Agatston scores it under-weights the long tail.
- Romano 2006 thresholds: 0.147 (small), 0.330 (medium), 0.474 (large). We adopt the "robust discriminator" threshold |delta| >= 0.20 as a conservative "small-to-medium" floor.

### Mann-Whitney over Welch's t

- Mann-Whitney is rank-based, distribution-free, and the natural significance companion to Cliff's delta. The two statistics share the rank-comparison view.
- Welch's t on z-scored data is sensitive to the same Gaussian-assumption failures as Cohen's d.

### FDR-BH across 41 comparisons per (cohort x partition)

- We compute Cliff's delta for 28 robust representatives + 13 spatial-PCA inputs = 41 features per (cohort x partition x cluster) cell. At alpha = 0.05 with no correction we expect ~2 false positives by chance. The signature_features.csv would silently contain spurious discriminators.
- Benjamini-Hochberg controls the false-discovery rate at alpha = 0.05 across the bundle. We use `statsmodels.stats.multitest.multipletests(method='fdr_bh')`.
- FDR-BH is applied PER (cohort x partition) bundle, not globally across all bundles. Each cohort's discriminator set is independent.

### Biological sanity check rationale

- The two spatial GMM k=2 clusters are labelled "focal" / "diffuse" by D026's deterministic rule (lower median `n_calcified_arteries` = focal). If a pipeline regression were to put a soft-tissue mask into the "focal" cluster, max_hu_global in that cluster would collapse to noise-floor (mean HU about -264 for soft tissue) far below the 130 HU calcium definition.
- The 130 HU absolute floor is the IBSI / Agatston standard for calcium identification and is therefore an objective, non-arbitrary threshold.
- A relative-ratio rule (focal >= 0.9 * diffuse) fails on this cohort because focal disease in COCA has lower peak HU (387) than diffuse (744) - a real biological pattern, not a pipeline bug. We raise only on the absolute floor; the ratio is a logged warning so unexpected biology is visible without blocking the pipeline.
- Failing loud at the absolute floor is the desired behaviour.

### Label balance threshold

- The Hennig spatial-only k=2 medians are 0.85-0.92 across cohorts (stage 6 Finding 3); this only holds when both clusters are reasonably populated. Below ~15% minority we cannot claim phenotype-level conclusions; the small cluster's IQRs become uninformative.
- 15% is conservative relative to Hennig's published recommendation (typically 25% for "well-supported"). We pick 15% because COCA's enrichment pattern naturally pushes one mode small; rejecting the cohort at 15% would over-reject.

## Alternatives considered

- **Cohen's d on z-scored data**: rejected for the reasons above; sentinel-prone density tiers and Agatston skew break the Gaussian assumption.
- **Bonferroni correction**: rejected as too conservative for radiomics-style exploratory characterisation. FDR-BH is the standard.
- **Per-cluster vs (everyone-else) instead of focal-vs-diffuse only**: for the spatial k=2 partition the two are identical. For the burden k=3 partition we explicitly use "this-cluster-vs-(other-two-combined)" because pairwise comparisons would multiply the FDR adjustment unnecessarily.
- **Permutation-based effect-size CIs**: under consideration for the paper but out of scope for stage 7. Reproducibility is via fixed random_state on bootstrap helpers if needed for figure CIs.

## Verified by

- `tests/analyse/test_profiles.py` covering: Cliff's delta against known cases (identical, disjoint, half-overlap); Mann-Whitney against scipy reference values; FDR-BH against statsmodels reference; biological sanity check raises on simulated soft-tissue focal cluster; label balance raises on 10/90 split.
