# D036 Detectability floor by synthetic cluster injection

**Date**: 2026-10-01
**Stage**: discover (validation of D021)
**Status**: Active. Written and fixed before any planted run was executed.
**Module**: `src/predict/discover/detectability.py`, `scripts/probe_detectability_floor.py`, `submit_detectability.sh`

## Question

Finding 1 says COCA forms a burden continuum with no discrete phenotypes. Without knowing what the pipeline can detect, "no clusters exist" and "the pipeline could not have found them" are observationally identical. D036 measures, on this cohort's own feature matrix, the smallest separation at which planted clusters are recovered.

## Design

- **Matrix**: the gated representative features of the D033 gate (`outputs/06_reduce/`; 440 x 35 when first run), z-scored. The gate, ComBat and redundancy steps are not rerun (planting happens after representative selection; stated as a limitation).
- **Planting**: k groups of random patients (independent of the data) receive mean shifts on the vertices of a regular simplex in a random (k-1)-dimensional subspace, so every pair of groups is separated by `delta` background standard deviations along the planting axes. Then re-z-score, PCA at the config cumvar (D020), and the D021 steps below. Nothing in the pipeline is tuned to the planted signal.
- **Grid**: k_true 2, 3, 4; delta 0, 1, 2, 3, 4, 6; direction random or burden-orthogonal (orthogonal to the real PC1 loading vector); proportions balanced or imbalanced (1, 1/2, 1/4, ...); background real (the matrix itself, which already contains the continuum) or gaussian (N(0, real covariance), a positive control with no continuum); 10 replicates per setting. delta = 0 is the null.
- **Measured per replicate**: Hopkins on the PCs; gap statistic (kmeans and ward, k 1..12, Maitra-Ramler 1-SE-from-argmax rule as in D021) with B = 100 bootstraps; forced k_true kmeans labels: ARI against the planted labels and Hennig clusterboot (100 bootstraps, minimum per-cluster median Jaccard).
- **Bootstrap count**: B = 100 instead of D021's 500 for cost. Calibration: the setting (k 3, random, balanced, real) is rerun with B = 500 on identical plantings and the agreement of selected k and detection is reported.
- **Seeds**: `default_rng([k, direction, proportions, background, 10*delta, rep, 36])`.

## Rules (fixed before running)

- **Primary detection**: the kmeans gap statistic selects exactly k_true (the D021 discovery decision).
- **Secondary**: ward gap selects k_true; "recovered" = forced-k ARI >= 0.80 AND every cluster Hennig median Jaccard >= 0.75.
- **Floor**: per (background, direction, proportions, k_true), the smallest delta > 0 with a detection rate >= 0.80 (8 of 10). "none" if no grid delta reaches it.

## Interpretation (fixed before running)

1. **Real-background floor finite** (primary, random direction, balanced): Finding 1 is stated with the floor: "discrete subgroups of 2 to 4 groups separated by at least `delta*` SD would have been detected with at least 80% probability; none were found."
2. **Gaussian floor finite but real floor "none"**: the gap statistic cannot detect clusters on top of this continuum at any tested separation. Finding 1 must be weakened to "no structure the gap statistic can resolve on this manifold", and the forced-k recovery ("recovered") floor is reported as the operative detection limit instead.
3. **Gaussian floor "none"**: the pipeline itself fails the positive control; Finding 1's discreteness claim is withdrawn until the detector is fixed.
4. **delta = 0 rows**: on the real background they should reproduce the observed behaviour (gap k near k_max); on the gaussian background, k = 1 or small. A detection rate above 0.20 at delta = 0 for any k_true is reported as a false-positive problem.

Burden-orthogonal and imbalanced settings are reported for every outcome; the headline sentence uses the random-direction, balanced setting, and the worst case across settings is stated next to it.

## Relation to earlier decisions

D020 and D021 are unchanged and run as is. D036 adds evidence for Finding 1; it does not change any production output.

## Verified by

- `tests/discover/test_detectability.py`: simplex equidistance and centring, group sizes, orthonormal and PC1-orthogonal directions, separation in background-SD units, delta = 0 identity, z-score.
- Smoke run (`--smoke`) before the full job.
