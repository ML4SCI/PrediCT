# Decision Log

One markdown file per decision: `D###-short-slug.md`. Numbers are monotonic; superseded decisions are kept but marked **Superseded by D###**. Each decision file follows the template below.

## Template

```
# D### Short title

**Date**: YYYY-MM-DD
**Stage**: io | preprocess | features | stability | reduce | discover | analyse | validate
**Status**: Active | Superseded by D### | Deferred
**Module**: src/predict/<stage>/<module>.py

## Decision
One paragraph stating the rule, threshold, or convention.

## Rationale
Why this and not the alternatives. Cite dataset evidence or literature.

## Alternatives considered
- Option A: rejected because …
- Option B: rejected because …

## Verified by
Test file/function or empirical check.
```

## Index

| ID | Title | Stage | Status |
|----|-------|-------|--------|
| [D001](D001-z-coordinate-matching.md) | Z-coordinate matching as primary ROI→slice mapping | io / preprocess | Active |
| [D002](D002-xml-stat-roundtrip.md) | XML stat round-trip as preprocessing correctness gate | preprocess / validate | Active |
| [D003](D003-no-display-hu-window.md) | No display HU window in pipeline outputs | preprocess | Active |
| [D004](D004-cohort-exclusions-at-discovery.md) | Cohort exclusions applied at patient discovery | io | Active |
| [D005](D005-target-voxel-grid.md) | Target voxel grid 0.375 × 0.375 × 3.0 mm (native median in-plane spacing) | preprocess | Active |
| [D006](D006-multi-series-selection.md) | Multi-series DICOM folder reduction rule (largest slice count) | io | Active |
| [D007](D007-lesion-grouping-rule.md) | Lesion grouping rule (BFS, ≤5 mm in-plane, gap=1) | features | Active |
| [D008](D008-per-artery-mask.md) | Per-artery masks via filter-then-rasterise | features | Active |
| [D009](D009-pyradiomics-config.md) | PyRadiomics extractor configuration | features | Active |
| [D010](D010-low-burden-flag.md) | `low_burden_flag` for very-small-mask patients | features | Active |
| [D011](D011-agatston-thickness-helper.md) | Single Agatston thickness-correction helper | features | Active |
| [D012](D012-excluded-roi-ids-input.md) | `excluded_roi_ids` as explicit input to every feature path | features | Active |
| [D013](D013-icc-formulation-and-threshold.md) | ICC(3,1) absolute agreement, threshold 0.75 | stability | Active |
| [D014](D014-perturbation-set.md) | 14-perturbation set (rotations, translations, noise) | stability | Active |
| [D015](D015-cohort-subset-for-stability.md) | Full eligible cohort (PyRadiomics ok) for ICC computation | stability | Active |
| [D016](D016-geometric-feature-bypass.md) | Geometric features bypass empirical gate with ICC=1.0, tagged via `icc_source` | stability | Active |
| [D017](D017-sentinel-prone-feature-exclusion.md) | Drop 13 zero-inflated per-vessel features at stage 5 ingestion | reduce | Active |
| [D018](D018-density-tier-and-dense-calcium-handling.md) | Keep 16 density tiers (robust scaling at stage 5); binarise dense_calcium_count -> has_dense_calcium | reduce | Active |
| [D019](D019-stage5-preprocessing.md) | Stage 5 preprocessing: variance filter, derived-feature R^2 gate, ComBat on 6 texture, Yeo-Johnson + rank fallback on 19 sparse, global z-score | reduce | Active |
| [D020](D020-stage5-redundancy-and-pca.md) | Spearman r^2 > 0.75 hierarchical clustering (average linkage primary; ward / complete sensitivity); ICC-first representative selection; global PCA at 0.85 cumvar | reduce | Active |
| [D021](D021-stage5-cluster-discovery-and-validity.md) | Hopkins -> gap statistic 500 boots x 3 runs (full / burden-residualised / spatial-only) -> consensus -> forced k=3 -> kernel/bootstrap validity -> robust-cohort + 2 kernel-stratified sensitivity reruns | discover | Active (re-tagged from `reduce` on 2026-06-05 Phase B; number unchanged) |
| [D022](D022-multi-block-redundancy-clustering.md) | Multi-block Spearman r² > 0.75 redundancy clustering across 6 prospective blocks (burden, HU stats, density tier, spatial, texture, shape); single-matrix variant retained as sensitivity | reduce | Active (supersedes single-matrix portion of D020 for primary analysis) |
| [D023](D023-stage7-per-cluster-characterisation.md) | Per-cluster characterisation: median + IQR + Cliff's delta + Mann-Whitney with FDR-BH across 41 features; biological sanity check (focal median max_hu >= 130 HU); label balance check (minority class > 15%) | analyse | Active |
| [D024](D024-stage7-burden-orthogonality.md) | Burden orthogonality on agatston_total: Mann-Whitney + Levene + Cliff's delta with 3-level interpretation column (orthogonal / marginal / confounded); burden-stratified spatial profiles within each Agatston tertile | analyse | Active |
| [D025](D025-stage7-directional-hypotheses.md) | Six pre-specified one-sided Mann-Whitney directional hypotheses for focal vs diffuse; primary (>= 4 of 6 at FDR p < 0.05 in full) + secondary (>= 4 of 6 same direction in BOTH strata, direction only) verdict | analyse | Active |
| [D026](D026-stage7-monotonicity-classification.md) | Spearman rho + Kendall tau vs agatston_total for each of the 28 robust features; classify into burden_tracking (\|rho\| >= 0.5) / structure_tracking / spatial_tracking / mixed | analyse | Active |
| [D027](D027-stage7-cross-cohort-feature-consistency.md) | Feature-level cross-cohort consistency: direction in all 3 + FDR p < 0.05 in >= 2 of 3 + \|Cliff's delta\| >= 0.20 in all 3; partition ARI on shared pids preserved as separate evidence pillar (>= 0.80) | analyse | Active |
| [D028](D028-stage7-low-burden-sensitivity-output.md) | Robust-cohort (low_burden_flag = False) sensitivity as a separate output file, not extra rows in the main paper table | analyse | Active |
| [D028.1](D028.1-low-burden-sensitivity-verdict.md) | D028 amendment: the subset gets D024, the D025 primary rule and recomputed signatures, and a verdict (same as / different from the full cohort) | analyse | Active |
| [D029](D029-stage8-external-ge-holdout.md) | External GE-scanner holdout (pids 19, 28, 76, 77): apply frozen pipeline; skip ComBat (spatial-only phenotype is ComBat-free); descriptive only (N=4); D023 mapping on predicted_phenotype | validate | Active |
| [D030](D030-stage8-leave-k-out-cv.md) | 10-fold kernel-stratified LOO with full per-fold refit (D019 + D022 + D020 + D021 + GMM); PASS = median ARI ≥ T where T = 5th-percentile simulation at K=10% disagreement; raw GMM labels into ARI; D023 mapping on interpretive columns only | validate | Active |
| [D031](D031-stage8-cross-cohort-ari-consolidation.md) | Re-export stage 7 cross-cohort ARI with PASS verdict columns (ARI ≥ 0.80); pure bookkeeping for the stage 8 deliverable triad | validate | Active |
| [D033](D033-realistic-perturbation-set.md) | Realistic composite perturbation set, ICC(1,1) over 20 draws; primary stability gate, D014 retired from downstream | stability | Active (locked 2026-09-24) |
| [D034](D034-robustness-tiers.md) | Robustness tiers on the D033 gate (T1 primary to T5 excellent), plus ICC(3,1) check; exploratory | stability | Active (exploratory) |
| [D035](D035-combat-columns-from-gate.md) | ComBat harmonises every gated non-shape PyRadiomics feature (list derived from the gate; fixes a stale 6-name list) | reduce | Active |
| [D036](D036-detectability-floor.md) | Detectability floor by synthetic cluster injection (rules and interpretation fixed before running) | discover | Active |
| [D036.1](D036.1-detectability-amendment.md) | D036 amendment: deltas to 16, GMM arm, PC1-SD units (fixed before running) | discover | Active |
| [D037](D037-lesion-morphology-features.md) | Lesion clustering on 4 non-degenerate features (volume/area ratio was the constant slice thickness; area duplicated volume) | lesion exploratory | Active |
| [D038](D038-taxometric-continuum-test.md) | Taxometric CCFI test, dimensional vs categorical (closed: not usable on strongly skewed continua; never run on real data) | discover | Experimental, closed |
| [D021.1](D021.1-kernel-policy-and-statistic-corrections.md) | D021 amendment: kernel failure followed by a burden-adjusted test (confounded stops, explained_by_burden reported); Hopkins uniform queries use the nearest neighbour; gap reference in the input's principal axes; Hennig refit failures raise | discover | Active |
| [D033.1](D033.1-broken-draw-definition.md) | D033 amendment: a broken draw is only an empty / below-minimum perturbed mask or a NaN feature; any other exception is an error that stops the stage | stability | Active |
| [D039](D039-slice-geometry-check.md) | Selected series must be regular, contiguous and axial, else excluded at discovery (33, 135, 159, 388); discovery and loaders share one selection | io | Active |
| [D040](D040-vessel-gini-and-derived-reliability.md) | vessel_burden_gini over all 4 vessels (calcium-free included, per D019); stage-5 transformed features get measured D033 reliability (5b derived_icc.csv) | reduce | Active |
| [D041](D041-stage8-production-fit-parity-and-coverage.md) | Stage 8: production fit (stage 5 prep settings, stage 6 seed and GMM), exact replay incl. rank mapping, parity with stages 5-6, exact cohort and coverage, all-or-nothing publication; replaces D030.1's GMM settings | validate | Active |
| [D043](D043-stage9-chain-integrity.md) | Stage 9: one published chain (exploratory.partial + COMPLETE_stage9.json), termination-aware LAD results, balance on the compared population, stratum-level signature replication, no historical LAD reference, complete and parity-checked lesion stability | exploratory | Active |
| [D044](D044-stage10-figure-claims-and-publication.md) | Stage 10: figure text follows the files and decisions (no continuum claim, threshold-rule classes, superior-inferior C8 position, no fixed numbers); one published figure set with COMPLETE_stage10.json; stage 9 fingerprint narrowed to the lesion-class module | figures | Active |
| [D039.1](D039.1-in-plane-grid.md) | D039 amendment: one in-plane grid per series (PixelSpacing, matrix, x/y origin); pid 156 | io | Active |
