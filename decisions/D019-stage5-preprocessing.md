# D019 Stage 5 preprocessing pipeline

**Date**: 2026-06-04
**Stage**: reduce (stage 5)
**Status**: Active
**Module**: `scripts/06_reduce.py` (when built; depends on D017 + D018 transforms already locked)

## Decision

Stage 5 ingestion executes the following preprocessing pipeline on the gated feature set, in order. The output is a (422, n_features) numerical matrix ready for redundancy clustering (D020) and PCA.

```
input: outputs/05_icc/gated_features.csv (88 features after D013 gate)

1. Apply D017 drops (4 diffusivity + 9 RCA/LCx/LM distance)        88 -> 75
2. Apply D018 binarisation (dense_calcium_count -> has_dense_calcium) 75 -> 75
3. Optionally append derived features (high_density_fraction,
   vessel_burden_gini) conditional on R^2 < 0.95 non-redundancy check  75 -> 75 to 77
4. Variance filter: drop columns with sd < 0.01 on full cohort      n -> n'
5. ComBat harmonisation on the 6 PyRadiomics texture columns        n' -> n'
   covariates: kernel group in {Qr36d/2, I30f/3}
6. Yeo-Johnson transform on the 19 sparse columns
   (16 density tier + 3 LAD distance) with per-column skewness fallback
   to rank transform if abs(post-YJ skew) > 1.0                     n' -> n'
7. Global z-score on all n' columns                                 n' -> n'

output: (422, n') float matrix; n' is expected ~75 to ~77 depending on derived-feature acceptance
```

`has_dense_calcium` is binary (0/1) and bypasses Yeo-Johnson and rank-transform; it goes into the global z-score directly.

## Rationale

### Variance filter

Columns with sd < 0.01 contribute essentially no separation between patients and only add noise to PCA. The threshold matches `configs/default.yaml` `reduce.variance_threshold`. Should not trigger on our 75 features (we have already audited density-tier and distance variance), but kept as a safety net for the derived features.

### Derived features (high_density_fraction, vessel_burden_gini, count_volume_ratio decision)

Three candidates were considered:

| feature | formula | clinical | redundancy risk | decision |
|---|---|---|---|---|
| `high_density_fraction` | (n_rois_d3 + d4) / total tier counts | Hoori 2024 mineralisation maturity | partial overlap with 16 tier counts | accept if R^2 against linear combinations of existing features < 0.95 |
| `vessel_burden_gini` | gini across 4 per-vessel Agatston values | focal vs distributed disease | partial overlap with per-vessel Agatston | accept if R^2 < 0.95 |
| `count_volume_ratio` | lesion_count_total / volume_total_mm3 | many small vs few large | high overlap with two existing features | reject |

The R^2 non-redundancy gate is conservative: it allows a derived feature to enter only if it carries variance unexplained by linear combinations of features already in the matrix. This guards against the editorial bias of inserting hypothesis-driven features that PCA could discover itself.

All accepted derived features inherit `icc_source = invariant_by_construction` because they are pure functions of XML-derived inputs.

### ComBat harmonisation

The COCA cohort spans **two majority** acquisition kernels plus two
singleton-kernel patients:

- Qr36d/2 (SOMATOM Force) 235 patients (52%)
- I30f/3 (SOMATOM Definition Flash) 207 patients (46%)
- B35f 1 patient, I36f/3 1 patient (singletons)

ComBat needs >= 2 samples per batch (kernel group) to estimate the
within-batch variance. The orchestrator therefore **filters the eligible
cohort to the two majority kernels before matrix prep**, dropping the
2 singleton-kernel patients (if they were not already in the 22
PyRadiomics-skipped patients). The post-filter cohort size is 420 to 422
depending on whether the singletons had radiomics_status == "ok". The
filtered patients are documented in the matrix_prep log.

A defense-in-depth assertion inside ``combat_harmonise`` raises if any
singleton-kernel sample slips through. The orchestrator is the canonical
place to do the filter; the assertion is a safety net.

PyRadiomics texture features are kernel-dependent because texture statistics measure local intensity patterns that differ between reconstruction kernels (Mackin 2017, Orlhac 2019). The 14 PyRadiomics shape features are kernel-independent by construction (they only depend on the binary mask geometry, which is set by the radiologist's polygon, not by CT pixel values). The 68 canonical features are also kernel-independent (they read XML's frozen Max/Mean, never CT pixel values).

So ComBat is applied to the **6 PyRadiomics texture survivors only** with kernel group as covariate:

- `original_glrlm_RunLengthNonUniformity`
- `original_gldm_DependenceEntropy`
- `original_glszm_ZoneEntropy`
- `original_glrlm_GrayLevelNonUniformity`
- `original_gldm_GrayLevelNonUniformity`
- `original_firstorder_Range`

Verification: post-ComBat, scanner-explained variance (R^2 of linear regression of each harmonised column on the kernel indicator) must drop below 0.02 (a common acceptance threshold; Lin 2022 Table S3 uses the same). Pre vs post values written to `outputs/06_reduce/combat_audit.csv` for the audit trail.

### Yeo-Johnson transform with rank fallback

Sparse continuous columns (35-90% zeros) violate PCA's Gaussian assumption severely. Three transforms were considered:

- **Rank-based**: replace each value by its rank divided by N. Most aggressive; removes the zero point mass entirely. Loses count magnitude. Always normalises.
- **Median / IQR**: robust scaling. Preserves the original distribution shape, including the zero spike.
- **Yeo-Johnson**: Box-Cox extension supporting non-positive values. Fits a transform parameter lambda by maximum likelihood that compresses the right tail and stretches the zero mass.

Yeo-Johnson is the right default because it preserves continuous-regime ordering and fits PCA's Gaussian assumption, but on columns with 70%+ zero rates the fitted lambda may produce a near-degenerate transform that still leaves heavy skew. The fallback rule (per-column post-YJ check: abs(skewness) > 1.0 implies rank-transform) is defensive engineering: it lets Yeo-Johnson win on the columns where it converges and falls through to rank on columns where it does not.

The 19 columns subject to YJ + fallback:

- 16 density tier counts (`n_rois_d{1..4}_{lad,rca,lcx,lm}`)
- 3 LAD distance features (`inter_lesion_dist_mean_lad`, `inter_lesion_dist_max_lad`, `first_to_last_dist_lad`)

`has_dense_calcium` (binary 0/1) skips the YJ step and goes straight into global z-score.

### Global z-score, not per-block

Two scaling architectures were considered:

- **Per-block z-score**: apply z-score within each of 6 hand-defined blocks (burden, density, spatial, morphology, texture, per-vessel HU) before concatenating. Implicitly weights blocks equally regardless of block size.
- **Global z-score**: apply z-score across the full matrix. Variance allocates weight naturally; features with higher variance contribute more to PCA.

Global z-score is the principled default for unsupervised PCA. Per-block z-score is editorial weighting that presupposes the block partition is correct and that all blocks deserve equal voice; in the absence of strong prior reason to weight blocks equally, this is hypothesis-baking. We adopt global z-score and use the block structure only for stage 6 interpretation (grouping features by block when characterising clusters), not as a PCA preprocessing step.

## Alternatives considered

- **Per-block z-score** rejected as editorial weighting; see above.
- **Median / IQR scaling for sparse columns** rejected because it does not address skewness, only outlier robustness.
- **Rank transform for all sparse columns directly** rejected because it discards count magnitudes more aggressively than necessary; Yeo-Johnson with fallback is the lighter touch.
- **No harmonisation across kernels** rejected because the literature is clear that PyRadiomics texture features under different reconstruction kernels are systematically biased (Mackin 2017, Orlhac 2019). Stage 6 would then likely find a "scanner cluster" rather than a phenotype cluster.

## Verified by

- `outputs/06_reduce/preprocessing_log.json` will record per-step shape and the variance-filter / YJ-fallback / ComBat audit results.
- `outputs/06_reduce/combat_audit.csv` will record pre vs post kernel-explained variance for each of the 6 texture features (acceptance: post-ComBat R^2 < 0.02).
- Tests will assert: matrix shape after each step matches the documented dimensions; ComBat acceptance threshold is met; derived features that fail the R^2 < 0.95 gate are explicitly omitted from the matrix with a logged reason; all columns have abs(skewness) < 1.0 after the YJ + fallback step.
