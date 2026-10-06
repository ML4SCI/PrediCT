# D020 Stage 5 redundancy clustering and PCA

**Date**: 2026-06-04
**Stage**: reduce (stage 5)
**Status**: Active
**Module**: `scripts/06_reduce.py` (when built; consumes the D019 preprocessing output)

## Decision

After D019 preprocessing, the (422, n') matrix passes through:

```
1. Spearman correlation matrix on the n' columns (rank-based, robust to non-Gaussian)
2. Hierarchical clustering on (1 - r^2) distance
   linkage: average
   cut: elbow detection with min gap 0.05, fallback at distance = 0.25 (r^2 = 0.75)
3. Representative selection per cluster
   primary key:   ICC (D013 empirical or invariant_by_construction == 1.0)
   tiebreaker 1:  feature class priority (canonical > PyRadiomics)
   tiebreaker 2:  alphabetical for deterministic output
4. PCA on the representative-feature subset
   cumulative variance retained: 0.85 (primary), 0.90 (sensitivity probe)
   centering: already z-scored in D019
5. Output: PC matrix (422, k), loadings, explained variance, representative list

Sensitivity reruns (write secondary tables):
- ward linkage on the same Spearman matrix
- complete linkage on the same Spearman matrix
```

## Rationale

### Spearman, not Pearson

Stage 5 input contains many non-Gaussian features (Yeo-Johnson-transformed sparse columns may still be moderately skewed; binary `has_dense_calcium`; integer count features). Pearson correlation under-estimates monotone relationships between non-Gaussian variables and is sensitive to outliers. Spearman correlation captures monotone dependence regardless of distribution shape, which is the correct semantics for "do these features carry redundant information" in a heterogeneous feature set.

### Cut at r^2 > 0.75 (distance 0.25)

An initial exploration suggested r > 0.75 (i.e. r^2 > 0.5625 or distance > 0.44 in the more common 1-r metric). That is conservative; many features above r=0.75 are still partially independent.

We use the **r^2 > 0.75 (distance > 0.25)** cut on the (1 - r^2) distance, which corresponds to **r > 0.866**. This is stricter than r > 0.75 and matches the standard "tight redundancy" definition (e.g. Lin 2022 uses r > 0.85; Kolossvary 2025 uses r > 0.90).

The elbow detector tries to find a natural distance threshold by looking for a gap > 0.05 in the dendrogram heights. If no clean elbow exists, the cut falls back to distance = 0.25 (the literature threshold).

### Average linkage primary, ward / complete as sensitivity

- **Average linkage** (UPGMA) is the standard default for feature-redundancy clustering; it produces clusters whose pairwise distances average to the linkage value.
- **Ward linkage** minimises within-cluster variance; produces tighter clusters but is sensitive to outlier features.
- **Complete linkage** uses worst-case pairwise distance; produces compact clusters but can fragment large redundancy groups.

We commit to average as the primary report and run all three as a sensitivity table. The composition of representatives across the three linkages tells us whether the redundancy structure is stable to the linkage choice or whether the result is method-dependent.

### Representative selection: ICC first, canonical second, alphabetical third

Within a redundancy cluster, the published practice is to keep one representative feature. The selection criterion can be:

- highest ICC (most reliable across perturbations)
- highest variance (most informative in raw signal terms)
- lowest correlation with cluster mean (most independent)
- domain expertise (e.g. most clinically familiar)

We use a **layered selection**:

1. **Primary**: highest ICC value. Under D016, canonical features all have ICC = 1.0, so canonical features beat empirical PyRadiomics on ICC ties.
2. **Tiebreaker 1**: canonical features beat PyRadiomics on equal ICC. This is a clinical-interpretability tiebreaker: in stage 6 cluster characterisation, `agatston_lad` is more interpretable than `original_glcm_Contrast`.
3. **Tiebreaker 2**: alphabetical, for reproducibility.

This rule is deterministic and easy to audit.

### PCA cumulative variance 0.85, sensitivity probe 0.90

For our 75-feature matrix expected to reduce to ~30 to 50 representatives, 0.85 retains roughly 10 to 20 PCs and 0.90 retains roughly 15 to 25.

0.85 is the modal radiomics phenotyping choice (Lin 2022, Kolossvary 2025, Mackin 2015 all use 80 to 85%). 90% is also defensible. We lock 0.85 as primary, and write a sensitivity table at 0.90 to show whether downstream clustering is sensitive to the cutoff.

Rationale for not going higher: lower-variance PCs in our matrix are likely to capture residual noise rather than phenotype signal. At n = 422, the curse of dimensionality starts to bite in clustering algorithms when k-PCs grows past ~25.

### Global PCA, not block-based

We adopt a single PCA over the representative-feature subset, not a per-block PCA. Reasoning matches D019: per-block PCA implicitly weights blocks equally regardless of their information content, which is editorial. Global PCA lets variance allocate weight naturally. The block structure is used only for stage 6 interpretation (grouping PC loadings by block to characterise what each PC represents).

## Alternatives considered

- **Pearson correlation**: rejected because non-Gaussian features in our matrix would under-estimate redundancy and let near-duplicate columns escape the cut.
- **r > 0.75 (looser)** as primary cut: rejected as more permissive than the radiomics literature; would let weak redundancy through.
- **r > 0.95 (very strict)** as primary cut: rejected as too permissive; under our small representative count this would leave many near-redundant columns.
- **Random representative within a cluster** (no rule): rejected; non-deterministic, audit-hostile.
- **Per-block PCA with concatenation**: rejected as editorial weighting (see D019).
- **0.80 cumvar cap**: rejected because at n = 422 we can afford the extra ~3-5 PCs and the variance retention gain reduces information loss.
- **0.95 cumvar cap**: rejected as too many PCs for stable downstream clustering.

## Verified by

- Tests will assert that the representative-feature list count matches the cluster count of the average-linkage cut at distance 0.25; that ward / complete sensitivity counts are within reasonable bounds of the primary; that the representative selection rule is deterministic (rerun gives the same names); that PCA cumulative variance at the kept PC count is between 0.85 and 0.86 (exact value depends on data).
- `outputs/06_reduce/redundancy_clusters.csv` lists each input column, its cluster id, the representative chosen, and the selection reason (ICC value, tiebreaker hit).
- `outputs/06_reduce/pca_loadings.csv` exposes the per-feature PC loadings for inspection.
- `outputs/06_reduce/representative_features.csv` is the input list for stage 6.
