# D021 Stage 5 cluster discovery, validity checks, and sensitivity cohorts

**Date**: 2026-06-04
**Stage**: discover (stage 6)
**Status**: Active
**Module**: `scripts/07_discover.py`; consumes `outputs/06_reduce/{prepared_matrix,pca_scores}.csv` from D020.

## Decision

After D020 PCA, the (422, k) PC matrix passes through cluster discovery, validity, and sensitivity:

```
A. Cluster tendency
   1. Hopkins statistic on the PC matrix (10% sample = 42 patients)
      threshold: > 0.65 (cluster tendency present)
                 0.55 to 0.65 (ambiguous, report both clustering outcomes)
                 < 0.55 (no cluster tendency, gap statistic expected to return k=1)

B. Cluster number selection (gap statistic, 3 runs)
   Run 1: full representative-feature space (post-D020)
   Run 2: burden-residualised space
          (regress out log(agatston_total + 1) from every feature first)
   Run 3: spatial-distribution-only space
          (only the surviving spatial features from D017-corrected Block C:
           lesion_count x 4 + total, n_calcified_arteries, gini_lesion_volume,
           dist_from_top_max/mean, center_of_mass_z, LAD distance x 3)

   For each run:
     - bootstrap n = 500 (Tibshirani 2001 standard)
     - k range: 1 to 12 (extended from 1..8 after smoke runs showed
       monotonic gap curve through k=8 with no plateau; we extend to 12 to
       confirm continuum structure persists at higher k or detect a
       plateau if it exists)
     - null reference: PCA-uniform
     - report selected k and the gap curve

   k-selection rule: **smallest k within one standard error of the maximum
   gap value** (Maitra and Ramler 2010, also the default in R's factoextra
   package). This is a robust variant of Tibshirani 2001's original
   "smallest k where gap does not significantly decrease" rule, which
   incorrectly selects k = 1 when the gap curve is non-monotonic at low k.
   The non-monotonic case arises routinely on highly-separated clusters
   because forcing the data into k < k_true clusters produces a
   within-cluster dispersion larger than the uniform reference, making gap
   negative at low k. The classic rule then matches gap(1) <= gap(2) and
   stops at k = 1. The 1-SE-from-argmax variant selects the smallest k
   whose gap is within one standard error of the maximum, which gives the
   true k_true for clustered data and k = 1 for genuinely uniform data
   (where the entire gap curve is flat within 1 SE).

C. Algorithms
   k-means (centroid), Ward linkage (linkage), Gaussian mixture (model-based)
   spectral clustering dropped: unjustified bandwidth (RBF gamma) on this problem

D. Consensus clustering
   n subsamples = 100
   subsample fraction = 0.80
   cluster instability index (CI) threshold for "stable cluster" = 0.60 (Monti 2003)
   k range: 2 to 8

E. Forced k = 3 characterisation (descriptive only, NOT phenotype discovery)
   Run k-means, Ward, and GMM at k = 3
   Crosstab cluster labels against:
     - Agatston tertile (burden axis check)
     - Agatston category (low / mid / high, finer)
     - n_calcified_arteries (multi-vessel count)
     - kernel (kernel confounder check)
     - low_burden_flag
     - high_density_fraction tertile (Hoori 2024 mineralisation maturity)
     - n_rois_d4_total tertile (high-density-tier ROI burden)
   Cluster medians of all features as `cluster_profiles.csv`.

F. Validity checks
   - Kernel chi-square: chi-square test on cluster x kernel contingency table for
     each algorithm. p < 0.05 => harmonisation failed; investigate.
   - Bootstrap stability (Hennig 2007 clusterboot): Jaccard similarity per cluster
     across 100 bootstrap resamples. Median Jaccard >= 0.75 => stable cluster.
   - Pre vs post ComBat scanner-explained variance (computed at D019); cited here
     for cross-reference.

G. PCA visualisation
   PC1 x PC2 scatter coloured by:
     - Agatston category (burden gradient check; PC1 must show clear gradient)
     - kernel (must NOT separate by kernel post-ComBat)
     - low_burden_flag (low-burden subset should be peripheral / dispersed)

H. Sensitivity cohorts
   1. Robust cohort: full pipeline rerun on patients with
      low_burden_flag == False (~280 of 422). Compare cluster solutions
      with the full cohort via Adjusted Rand Index (ARI) at each k.
      ARI < 0.5 between full and robust at the "preferred" k => clusters
      are low-burden-dependent, flag as a caveat.
   2. **Kernel-stratified reruns**: full pipeline rerun separately on each
      of the two ComBat-eligible kernel groups (Qr36d/2 ~220 patients;
      I30f/3 ~200 patients). ComBat is a no-op within a single-kernel
      cohort (no batches). The script is the same orchestrator
      ``scripts/06_reduce.py`` with the ``--kernel-filter`` flag. Outputs
      land under ``outputs/06_reduce/stratified_<kernel>/``. The intent
      is to verify the continuum-vs-clusters finding replicates
      independently within each scanner population, ruling out
      patient-population selection bias as the source of the kernel-
      cluster chi-square association at forced k=3 observed on the full
      cohort.

I. Outputs (under outputs/06_reduce/)
   preprocessing_log.json    D019 step audit
   combat_audit.csv          D019 ComBat pre/post scanner R^2
   redundancy_clusters.csv   D020 feature -> cluster id + representative chosen
   representative_features.csv  D020 surviving feature list (input to PCA)
   pca_loadings.csv          D020 per-feature PC loadings
   pca_explained_variance.csv  D020 per-PC variance explained
   hopkins.json              D021 cluster tendency
   gap_statistic.csv         D021 three runs x k=1..8 x gap value
   consensus_matrices.npz    D021 per-(algorithm, k) consensus matrices
   cluster_labels.csv        D021 per-(algorithm, k) cluster assignments
   cluster_profiles.csv      D021 cluster medians (forced k=3) for each algo
   forced_k3_crosstabs.csv   D021 crosstabs (cluster x covariate)
   kernel_confounder.json    D021 chi-square p-values per algorithm
   bootstrap_stability.csv   D021 Hennig Jaccard per cluster
   sensitivity_robust/       D021 same outputs on the 280-patient subset
   plots/                    D021 PCA scatters + dendrograms + gap curves
```

## Rationale

### Hopkins statistic first

Hopkins (1954) is a 2-second computation that estimates cluster tendency via the ratio of nearest-neighbour distances in real data vs uniform random data. Values near 0.5 indicate random / no cluster tendency; values near 1.0 indicate strong clustering; values near 0 indicate regular spacing (also no clustering). The 0.65 threshold for "clusterable" follows Banerjee 2004 and is conservative; the ambiguous band of 0.55 to 0.65 is reported with both clustering outcomes per the standard radiomics phenotyping convention.

Hopkins is run BEFORE gap statistic to short-circuit the 500-bootstrap cost if there is no cluster tendency. It does not over-rule gap; it is a low-cost guard.

### Three-run gap statistic

The three-run design is essential for an unsupervised phenotyping paper:

- **Run 1 (full)** answers "is there ANY discrete structure?" The default test.
- **Run 2 (burden-residualised)** answers "is there phenotype structure BEYOND the burden axis?" This is the most important run for the question of structure beyond burden.
- **Run 3 (spatial-only)** answers "is there spatial-distribution structure on its own?" This isolates the spotty / diffuse / dense hypothesis from burden and texture confounders.

`log(agatston_total + 1)` (rather than raw agatston_total) is the standard burden regressor because Agatston is right-skewed across nearly 3 orders of magnitude (1 to 3000+) and a linear regressor on raw values would overweight the high tail. The +1 protects against log(0).

n_bootstrap = 500 is the Tibshirani 2001 reference value. Compute cost ~4 to 8 minutes per algorithm-k pair; total under 1 hour for all three runs.

### Algorithms: k-means, Ward, GMM

These three cover three different mathematical families and a healthy slice of unsupervised clustering hypotheses:

- **k-means** assumes spherical equal-variance clusters; centroid-based.
- **Ward linkage** assumes within-cluster variance minimisation; agglomerative.
- **GMM** assumes Gaussian-mixture components; can handle elongated / unequal-variance clusters.

If all three converge on the same k and produce similar Adjusted Rand Index (>0.8) cluster solutions, the result is robust to algorithmic choice. If they disagree, the data does not have a single clean cluster structure.

**Spectral clustering** was dropped because its only meaningful hyperparameter (the RBF kernel bandwidth gamma) cannot be principledly chosen for this problem without external tuning data; we do not have a held-out clustering test set. Spectral is also conceptually similar to k-means on a Laplacian eigenmap and adds little marginal information beyond the three above.

### Consensus clustering

Monti 2003 consensus clustering is the standard unsupervised method for assessing cluster stability across resamples. The 100 subsample default with 80% subsample fraction is the published convention. CI threshold 0.60 separates "stable" from "unstable" clusters. Outputs include the per-(algorithm, k) consensus matrices for inspection.

### Forced k = 3 characterisation as descriptive output

Whatever gap statistic returns, we additionally run k-means / Ward / GMM at k=3 and produce cluster-median profiles + crosstabs. This is INTERPRETATION tooling, NOT phenotype discovery. The framing is critical:

- If gap returns k=1, the forced k=3 result is descriptive of arbitrary 3-way partition of a continuum.
- If gap returns k=3, the forced k=3 result describes the discovered phenotypes.
- If gap returns k != 3, the forced k=3 is reported as a one-line note ("clustering at the hypothesised k=3 produces the following profiles, but gap statistic selects k=X").

This discipline is essential to avoid presenting a forced cluster solution as if it were a discovered phenotype.

### Validity checks

The three validity checks fail hard or soft:

- **Kernel chi-square**: a hard fail. If p < 0.05 on cluster x kernel, ComBat did not adequately harmonise the texture features and the cluster solution is scanner-confounded. Re-investigate D019 ComBat.
- **Hennig bootstrap stability (clusterboot)**: median Jaccard < 0.75 per cluster across 100 bootstraps means the cluster is unstable to perturbation of the cohort. A cluster with low Jaccard is flagged in the writeup; not necessarily a re-design trigger, but a known fragility to report.
- **Pre / post ComBat scanner R^2**: should be < 0.02 post-ComBat (cited from D019 audit); if not, we do not even get to the clustering step.

### Sensitivity on the robust cohort

The 142 low-burden patients (mask_voxels < 100) are the most likely source of phenotyping noise: their PyRadiomics features are numerically defined but biologically suspect, and their canonical features sit on the sparse end of the distribution. Rerunning the full stage 5 pipeline on the 280-patient robust cohort and comparing via Adjusted Rand Index (ARI) tests whether the phenotype structure is driven by these borderline patients or by the broader cohort.

- ARI > 0.75 between full and robust cohorts at the preferred k => clusters are stable to low-burden inclusion.
- ARI between 0.5 and 0.75 => moderate dependence; report as a caveat.
- ARI < 0.5 => clusters are low-burden-dependent; investigate.

GE-scanner holdout (N=4 from D004) is deferred to stage 7 / validate (planned external check), not run here.

## Alternatives considered

- **Spectral clustering kept**: rejected because the unjustified bandwidth hyperparameter would produce arbitrary results.
- **Single-algorithm clustering (k-means only)**: rejected because robustness-across-algorithms is essential for an unsupervised paper.
- **n_bootstrap = 50 (existing config)**: rejected as below Tibshirani 2001's recommendation; gap-curve estimates would be too noisy at the boundary k values.
- **CI threshold 0.50 or 0.70**: rejected; 0.60 is Monti 2003's threshold.
- **No forced k=3 characterisation**: rejected because the spotty/diffuse/dense hypothesis is the explicit basis of the entire project and needs a descriptive report regardless of what gap selects.
- **Robust cohort defined differently (e.g. Agatston > 100)**: rejected; `low_burden_flag == False` aligns with D010.

## Verified by

- Tests will assert: Hopkins is computed on the PC matrix and the threshold band is logged; gap statistic produces 3 runs x 8 k values for each algorithm; consensus matrices are symmetric and in [0, 1]; forced k=3 produces 3 cluster labels with cluster size >= 5; kernel chi-square p-values are present; ARI between full and robust is in [-1, 1].
- All outputs above are written to `outputs/06_reduce/` with the specified names.
- The kernel confounder check fails hard (script exits non-zero) if p < 0.05; investigation step is documented in the stage doc.
- The script writes `run_header.json` with git commit, params SHA, library versions, and the random seeds used for each stochastic step (k-means init, gap bootstrap, consensus resampling, Hennig bootstrap).
