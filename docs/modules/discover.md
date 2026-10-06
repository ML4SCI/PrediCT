# Stage 6, discover (cluster discovery, validity, sensitivity)

## What this stage does

Reads the seam files produced by `scripts/06_reduce.py` and runs all of D021
on the PC matrix:

1. **Hopkins clusterability** on `X_full` (verdict: clustered, random,
   ambiguous).
2. **Gap statistic** on 3 algorithms x 3 feature spaces, 500 bootstraps each,
   with the Maitra-Ramler 2010 1-SE-from-argmax k-selection rule.
   - feature spaces: full PC matrix; burden-residualised PC matrix (against
     log(agatston_total + 1)); spatial-only feature subset (re-PCA'd from
     the 13 spatial features in `prepared_matrix.csv`).
   - algorithms: k-means, Ward, Gaussian mixture.
3. **Monti consensus** at gap-selected k for each algorithm (100 subsamples,
   80% fraction).
4. **Forced k=3 characterisation**: cluster sizes per algorithm + crosstabs
   against burden tertile, kernel, low_burden_flag, category. Descriptive
   only, NOT a discovery claim.
5. **Validity checks**:
   - Kernel chi-square confounder test (HARD FAIL at p < 0.05).
   - Hennig clusterboot bootstrap stability at forced k=3 (median Jaccard
     per cluster).
   - Hennig stability on spatial-only x GMM x k=2 (the Finding-3 phenotype
     probe).
6. **ARI cross-algorithm** at forced k=3.

The script seam is
`outputs/06_reduce/{pca_scores.npy, prepared_matrix.csv, cohort_metadata.csv}`
to `scripts/07_discover.py`.

## Module list

- `src/predict/discover/clusterability.py`  Hopkins statistic (real queries
  use their 2nd neighbour, uniform queries their 1st; D021.1);
  `assess_clusterability` returns a verdict.
- `src/predict/discover/cluster_discovery.py`  `fit_cluster` (kmeans / ward / gmm),
  `gap_statistic` with 1-SE-from-argmax k selection, `monti_consensus`,
  `burden_residualise`, within-cluster dispersion `W`. joblib-parallel in
  the gap and consensus loops.
- `src/predict/discover/validity.py`  `kernel_chi_square`, `hennig_clusterboot`,
  `ari`, `ari_on_shared_pids`.
- `scripts/07_discover.py`  the orchestrator. Reads the seam, runs all of D021,
  writes outputs back into the same cohort directory.

## Why these design choices

### Hopkins: real queries skip themselves, uniform queries do not (D021.1)

Hopkins compares nearest-neighbour distances of points sampled from the data
with those of uniform points in the data's bounding box, H = sum(w) /
(sum(u) + sum(w)), w the uniform-query distances, u the real-query distances.

- A real query is a row of X, so its nearest neighbour in X is itself at
  distance 0: it uses the 2nd neighbour.
- A uniform query is not in X: it uses its nearest neighbour (as
  `pyclustertend` does).

Letting the uniform queries also skip their nearest neighbour inflates H on
random data (mean 0.667 over 100 one-dimensional uniform datasets; smaller
bias in higher dimensions). Plain distances are kept (the
d-th power form of Hopkins 1954 is unstable at d >= 5). The 0.55 / 0.65
bands are unchanged; tests pin the corrected H near 0.5 on uniform data in
1 and 5 dimensions.

### Gap reference box in the input's own principal axes (D021.1)

Tibshirani's PCA-uniform null samples uniformly in the box spanned by the
data's principal axes. Stage 5 PC scores already are principal axes, but the
burden-residualised scores are not: regressing burden out of each PC makes
the columns correlated. `gap_statistic` now rotates every input into its own
principal axes (SVD of the centred input, all dimensions kept), samples the
box there and maps the samples back, so the reference is right for every
feature space. In a synthetic regression example a per-column box selects
k = 6 and the principal-axes box k = 1, with identical pairwise distances.

### Kernel confounding policy (D021.1)

The cluster x kernel chi-square at the forced k is computed per algorithm.
If it fails (p < 0.05), the likelihood-ratio test of kernel ~ cluster +
log1p(Agatston) against kernel ~ log1p(Agatston) decides: `explained_by_burden`
(adjusted p >= 0.05) is recorded and reported, `confounded` (adjusted
p < 0.05) stops stage 6 with exit 1 and nothing published. Single-kernel
cohorts record `not_applicable_single_kernel`.

### Hennig refit failures raise (D021.1)

A failed bootstrap refit raises. Scoring it as Jaccard 0 would make a
software failure read as cluster instability.

### Reading of gap results (D021.1)

All gap-statistic results are descriptive outputs of the specified
procedure. Selected k, including k = 1 or selection near the upper
boundary, is not treated as evidence establishing or excluding discrete
subtypes. Null-selection rates from the D036 / D036.1 probes (full space,
implementation before D021.1; see the decision) are methodological context
and are not transferred to uncalibrated feature spaces or the corrected
implementation.

### Publication, completion records and smoke runs

Stage 6 writes to `<cohort-dir>.partial/` with copies of the stage 5 files
listed in `COMPLETE_stage5.json`, then writes `COMPLETE_stage6.json` (stage 5
record hash, stage 6 code/config fingerprint, settings, input and output
hashes, spatial features used, kernel status) and swaps the folder in
(`predict.publish.publish_keeping_subdirs`, which keeps other cohorts'
sub-folders). A failed run leaves the last complete result in place.
Stages 7, 8, the focal/diffuse verification and the figures call
`predict.discover.stage6.require_stage6_complete`, which refuses a missing
record, a record from other stage 5 inputs or other stage 6 code/config, and
any changed output. The burden-residualised and spatial-only analyses are
required: a missing input stops the stage instead of skipping the analysis.
Any non-production setting (bootstraps, k range, forced k, seed) makes a
smoke run, written to `outputs/07_discover_smoke/<cohort>/` and marked
PARTIAL_SMOKE_DO_NOT_USE.

### Gap statistic with 1-SE-from-argmax (Maitra-Ramler 2010), NOT Tibshirani 2001

Tibshirani's classic rule "smallest k where gap(k) >= gap(k+1) - sk(k+1)"
fails on data with strong separation: at low k the gap is negative (within-
cluster dispersion exceeds uniform reference). gap(1) and gap(2) are both
negative and roughly equal, so the rule picks k=1 even when the true elbow
is at k=3. The R factoextra default (Maitra-Ramler 1-SE-from-argmax)
avoids that failure and recovers k on the well-separated blobs of the unit
tests. It is not reliable beyond that: on this cohort's geometry the gap
arms failed the planted-group positive control up to delta 16 (D036.1), and
in a thin-band scenario (300 points uniform along a line, so the data match
the uniform reference) it selected k = 1 in only 8 of 20 null datasets
(D021.1). Gap results are descriptive outputs only (see "Reading of gap
results" above).

### k_range extended to 1..12

With k=1..8, 7 of 9 algorithm x space combinations selected the boundary,
so the range is 1..12. Extending k_max to 20 (`scripts/probe_kmax_extension.py`,
implementation before D021.1) moved the selected k upward in 26 of 27
cohort x algorithm x space combinations (median 11 to 18): the selected k
tracks whatever ceiling is imposed. D036.1 found the same tracking on the
real matrix with up to four planted groups carrying 50% of the variance, so
selection at or near the boundary is not evidence for a continuum or for
discrete subtypes, and a curve that keeps rising to k_max is not read as
either.

### Algorithms (kmeans + Ward + GMM, no spectral)

Spectral clustering requires a similarity-kernel bandwidth that is not
theoretically justifiable on this feature space. D021 uses three algorithms
whose only hyperparameter is k.

### Spatial-only k=2 as the Finding-3 probe

The spatial-only subspace (13 features describing per-vessel lesion count
and inter-lesion geometry) gives a reproducible GMM k=2 partition: Hennig
median Jaccard 0.84 to 0.94 across all three cohorts (full, Qr36d/2,
I30f/3). k=2 was chosen by Hennig stability, never by the
gap statistic. Reproducible is not burden-independent: the two groups
differ strongly in total burden in all three cohorts (D024 confounded), so
the partition is read as a burden dichotomy (focal = low burden, diffuse =
high burden), not as a burden-independent spatial phenotype. Each
07_discover.py run writes the spatial GMM k=2 labels to
`cluster_labels_spatial_k2.csv` for downstream stage 7.

### Forced k=3 is descriptive, NOT discovery

Forced k=3 partitions are useful for sanity (e.g., kernel chi-square at the
same k across algorithms, ARI between algorithms at a common k). They are
NOT a phenotype claim: k=3 is fixed in advance, not discovered, and the
gap statistic is descriptive only and can neither support nor exclude it
(D021.1). The script's forced_k_crosstabs.csv exists to characterise the
forced partition against burden / kernel / category covariates, not to
support phenotype existence.

## Module contracts

```python
# clusterability.assess_clusterability
def assess_clusterability(
    X: np.ndarray,
    sample_frac: float = 0.10,
    threshold: float = 0.65,
    ambiguous_band: tuple[float, float] = (0.55, 0.65),
    random_state: int = 42,
) -> HopkinsResult:
    """Returns H, verdict in {clustered, ambiguous, random}, sample size, seed."""

# cluster_discovery.gap_statistic
def gap_statistic(
    X: np.ndarray,
    algorithm: ClusterAlgorithm,
    k_range: tuple[int, ...] = (1, 2, ..., 12),
    n_bootstrap: int = 500,
    null_reference: str = "pca_bounding_box",
    random_state: int = 42,
    n_jobs: int = 1,
) -> GapStatisticResult:
    """Returns log_Wk_observed, log_Wk_ref_mean/std, gap_values, sk_values,
    selected_k via Maitra-Ramler 1-SE-from-argmax."""

# cluster_discovery.monti_consensus
def monti_consensus(
    X: np.ndarray,
    k: int,
    algorithm: ClusterAlgorithm,
    n_subsamples: int = 100,
    subsample_frac: float = 0.80,
    random_state: int = 42,
    n_jobs: int = 1,
) -> ConsensusResult:
    """Returns (n, n) consensus matrix + PAC score."""

# validity.hennig_clusterboot
def hennig_clusterboot(
    X: np.ndarray, labels: np.ndarray, k: int, algorithm: ClusterAlgorithm,
    n_bootstrap: int = 100, threshold: float = 0.75, random_state: int = 42,
    n_jobs: int = 1,
) -> HennigStabilityResult:
    """Returns per-cluster jaccard_median, jaccard_mean, stable_mask."""
```

## Decisions

- D021 - cluster discovery, validity, and sensitivity cohorts.
- D021.1 - kernel policy; corrected Hopkins, gap reference box and Hennig
  failure handling.

## Tests

`tests/discover/`:

| Module | Test file |
|---|---|
| clusterability | `test_clusterability.py` |
| cluster_discovery (gap, monti, fit_cluster, burden_residualise, W) | `test_cluster_discovery.py` |
| validity (chi-square, Hennig, ARI, Jaccard) | `test_validity.py` |
| detectability probe helpers (D036) | `test_detectability.py` |
| taxometric test (D038) | `test_taxometric.py` |
| stage 6 publication safeguards | `test_stage6_script.py` |

Key tests: parallelism byte-identity (gap, monti, Hennig produce identical
results at n_jobs=1 vs n_jobs=4), PC sign invariance (already in pca tests
but the seam relies on it), Hopkins H near 0.5 on uniform random data in 1
and 5 dimensions and > 0.7 on three-blob synthetic data.

## Results on the analysis cohort

Full cohort 436, Qr36d/2 231, I30f/3 205; production settings (gap 500,
consensus and Hennig 100 bootstraps).

| | Full | Qr36d/2 | I30f/3 |
|---|---|---|---|
| Hopkins H | .754 | .714 | .705 |
| Gap selected k, full space (kmeans / ward / GMM) | 11 / 12 / 12 | 11 / 11 / 10 | 12 / 12 / 12 |
| Gap selected k, burden-residualised | 12 / 11 / 8 | 10 / 10 / 11 | 11 / 11 / 12 |
| Gap selected k, spatial-only | 10 / 11 / 3 | 11 / 11 / 8 | 7 / 7 / 8 |
| Spatial-only GMM k=2, Hennig median Jaccard | .890 / .888 | .935 / .935 | .841 / .860 |

- Gap results are descriptive (D021.1); most curves are still rising at
  k_max.
- Forced k=3: Hennig-stable clusters kmeans 3/3 (full), 3/3 (Qr36d/2), 1/3
  (I30f/3); ward 1/3 in each cohort; GMM 0/3 in each. ARI between
  algorithms .26 to .72. Consensus PAC .14 to .26.
- Kernel association at forced k=3 (full cohort): kmeans chi-square p .033,
  burden-adjusted p .115, so `explained_by_burden` (reported, D021.1); ward
  p .190 and GMM p .139 pass.
- The spatial-only k=2 partition replicates within each kernel stratum, so
  it is not driven by the kernel difference between strata; it is a burden
  dichotomy (D024 confounded, stage 7).

## Outputs

Published into `--cohort-dir` alongside the reduce-stage outputs (whole
folder swapped in after a complete run, with `COMPLETE_stage6.json`):

| File | Content |
|---|---|
| `hopkins.json` | H, verdict, sample size, seed |
| `gap_statistic.json` | full gap curves per (space, algo) |
| `gap_statistic_summary.csv` | selected k per (space, algo), plus gap values |
| `consensus_matrices.npz` | per-algorithm (n, n) consensus matrices |
| `consensus_summary.json` | PAC score per algorithm |
| `forced_k_characterisation.json` | cluster sizes per algorithm at forced k |
| `cluster_labels_forced.csv` | per-pid labels per algorithm at forced k |
| `cluster_labels_spatial_k2.csv` | per-pid GMM k=2 labels on spatial-only |
| `forced_k_crosstabs.csv` | cluster x covariate counts |
| `validity_checks.csv` | chi-square + Hennig (forced k AND spatial k=2) |
| `ari_across_algorithms.csv` | pairwise ARI at forced k=3 |
| `run_header_discover.json` | git hash, library versions, CLI args, SHA of seam files |
| `COMPLETE_stage6.json` | completion record checked by stages 7, 8 and the figures |

## How to run

```bash
# Production discovery on the full cohort (workers from PREDICT_N_JOBS or the scheduler)
python scripts/07_discover.py --cohort-dir outputs/06_reduce/

# Kernel-stratified cohorts (matching the strata produced by 06_reduce.py)
python scripts/07_discover.py --cohort-dir outputs/06_reduce/stratified_Qr36d_2/
python scripts/07_discover.py --cohort-dir outputs/06_reduce/stratified_I30f_3/

# Smoke (small bootstraps for ~2 min runtime; numerically noisy). Any
# non-production setting writes to outputs/07_discover_smoke/<cohort>/ and
# can never replace the cohort's published results.
python scripts/07_discover.py --cohort-dir outputs/06_reduce/ \
    --gap-bootstraps 20 --consensus-subsamples 20 --hennig-bootstraps 20
```

Wall-clock at production parameters: a few minutes per cohort on a 16-core node.

## Known limitations

- Forced k=3 is descriptive characterisation only; do NOT cite it as a
  discovery claim. The gap statistic is descriptive and can neither
  support nor exclude k=3.
- Hopkins H > 0.65 ("clustered") is not evidence of discrete phenotypes: a
  skewed burden axis raises H without separated groups. Neither Hopkins
  nor the gap curve establishes a continuum either (D021.1, D036.1, D038).
- The spatial-only Finding-3 partition uses GMM-specific k=2, chosen by
  Hennig stability, not by the gap statistic. It is reproducible across
  cohorts and is a burden dichotomy (D024 confounded).
- Hopkins on nearest-neighbour distances is sensitive to bit-level
  differences in `pca_scores.npy` across BLAS / numpy builds, most in the
  smaller kernel strata. Treat it as a supporting range, not a headline
  point estimate; the spatial-only k=2 Hennig medians are the more stable
  cluster-tendency summary.
