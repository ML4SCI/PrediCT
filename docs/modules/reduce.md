# Stage 5, reduce (matrix preparation, redundancy clustering, PCA)

## What this stage does

Transforms the stage-4-gated features into a low-dimensional PC matrix on the
ComBat-eligible patients, in three steps. Which gate it reads is set by
`stability.primary_gate` in `configs/default.yaml` and resolved by
`predict.config.gate_dir`: `realistic` = D033 (`outputs/05b_icc_realistic/`,
the primary gate), `classic` = D014 (`outputs/05_icc/`, supplementary; also the
layout of gate roots built by `scripts/make_gate_root.py`). Stage 8 (validate)
reads the gate through the same helper.

1. **D019 matrix preparation**: D017 sentinel drops (13 zero-inflated per-vessel features), D018 dense-calcium
   binarisation, conditional derived features (`high_density_fraction`,
   `vessel_burden_gini`) with an R^2 < 0.95 non-redundancy gate, variance filter
   (sd >= 0.01), ComBat on every gated non-shape PyRadiomics column (D035) with kernel
   covariate (target post-ComBat scanner R^2 < 0.02), Yeo-Johnson with
   rank-transform fallback on sparse columns, global z-score.
2. **D020 / D022 redundancy clustering**: Spearman r^2 > 0.75 clustering. The
   primary path (D022) runs multi-block over 6 prospectively-defined blocks
   (burden / HU statistics / density tier / spatial / texture / shape). The
   single-matrix variant (D020) is preserved as a sensitivity probe via
   `--block-mode single`. Representatives are chosen per cluster by ICC-first,
   canonical-over-PyRadiomics, alphabetical tiebreak.
3. **D020 PCA**: sklearn `PCA` (`svd_solver='full'`) on the representatives,
   sign-normalised so the dominant-loading feature is non-negative per PC,
   retained at 0.85 cumulative variance.

Cluster discovery, gap statistic, consensus clustering, forced-k
characterisation and Hennig validity are the **discover** stage; see
`docs/modules/discover.md`.

## Module list

- `src/predict/reduce/prepare_matrix.py`  D019 pipeline: drops, derived
  features, variance filter, ComBat, Yeo-Johnson + rank fallback, z-score.
  `MatrixPrepLog` records every step for audit.
- `src/predict/reduce/redundancy.py`  D020 single-matrix and D022 multi-block
  Spearman r^2 clustering, representative selection. 6 prospective
  `FeatureBlock` partitions in `DEFAULT_BLOCKS`.
- `src/predict/reduce/pca.py`  PCA with sign normalisation (`normalise_pc_signs`),
  family-tagged loadings table, `PcaResult` dataclass.
- `scripts/06_reduce.py`  the orchestrator. Reads stage 3 features + stage 4
  ICC, runs D019/D020/D022/PCA, writes the seam files for `07_discover.py`.

## Why these design choices

### D019 / D035 ComBat on PyRadiomics intensity features only (not on canonical or shape)

- Canonical features (Agatston, mass, volume, HU stats, density tiers, spatial,
  count, shape) read from XML's frozen `Max`/`Mean` plus the polygon vertices.
  No CT array indexing, so kernel cannot bias them. ComBat would be a no-op
  with overhead.
- Gated PyRadiomics firstorder and texture features DO index into the CT
  array and carry kernel effects. ComBat harmonises every one of them that
  the gate passes; the column list is derived from the gated set by
  `harmonisable_columns()`, so it cannot go stale when the gate changes
  (D035). Shape features read only the mask and pass through.

### D019 Yeo-Johnson with rank fallback

- 35-93% sentinel rates on LM density-tier bins make a pure z-score produce a
  non-Gaussian column that PCA misinterprets.
- Pure rank transform is too aggressive: it discards count magnitude in the
  continuous regime.
- Yeo-Johnson with per-column post-transform skewness check (abs > 1.0
  triggers rank fallback) gets the best of both; the sparse LM density tiers
  are the columns that fall back to rank.

### D022 multi-block over D020 single-matrix

In a calcium-positive cohort most texture and spatial features correlate
strongly with total burden, so a single-matrix Spearman clustering absorbs
them into the burden cluster and most representatives end up
burden-flavoured. The composition is the problem, not the count (numbers in
D022).

The multi-block partition runs Spearman clustering INDEPENDENTLY within 6
prospective blocks (burden, hu_statistics, density_tier, spatial, texture,
shape). The block boundaries are defined a priori from PyRadiomics IBSI
taxonomy, canonical feature provenance, and Lin 2022 / Hoori 2024, NOT from
any stage-5 finding. Representatives are then balanced across the blocks,
and the representative sets of the full and kernel-stratified cohorts
overlap far more than under the single matrix.

The single-matrix variant is retained as a sensitivity probe via
`--block-mode single`. The composition argument, not just the rep count, is
the reason multi-block is primary.

### D020 PC sign normalisation

PCA eigenvectors are sign-arbitrary. Without normalisation, "high PC1" can
mean opposite things on two runs of the same code on the same data, and
downstream clustering then assigns opposite labels. Convention: for each PC, flip the sign so the feature with the largest
absolute loading has a non-negative loading. Tested by
`test_fit_pca_sign_invariant_under_input_sign_flip`.

### D040 vessel Gini and measured reliability of transformed features

`vessel_burden_gini` is the Gini of the four per-vessel Agatston values with
calcium-free vessels included (0 = even, 0.75 = all burden in one vessel), as
D019 defines it. Dropping calcium-free vessels first would give single-vessel
patients (140 of 436 in the analysis cohort) the value of perfectly even
disease. The stage 7 helper imports the same function.

`has_dense_calcium`, `high_density_fraction` and `vessel_burden_gini` are
created here, so the gate never saw them. Stage 5b computes them on every
primary draw with `stage5_transformed_columns` and gates them with the D033
rule (`derived_icc.csv`). Stage 5 admits only passing ones and uses their
measured ICC (no assumed 1.0) when choosing representatives; with a D014
classic gate they stay invariant by construction.

### Run safeguards

- Every gated feature must be present in `features.csv`, or stage 5 stops.
- Each cohort writes to `<cohort>.partial/` and replaces the cohort's files
  only after the whole run succeeds, by an atomic folder swap
  (`predict.publish.publish_keeping_subdirs`: the current sub-cohort folders
  such as `stratified_*` are copied into the partial folder first, then the
  folder is swapped by two renames, so readers never see a mix). A failed
  run leaves the published folder untouched; the swap drops the old
  top-level files, including stage 6 outputs a stage 5 rerun invalidates.
- `COMPLETE_stage5.json` lists every stage 5 file with its hash and the hash
  of the `features.csv` the run read (checked unchanged at the end of the
  run). Stage 7 joins raw features.csv values to the stage 6 labels and
  refuses any other features.csv.
- `--block-mode single` (D020 sensitivity) writes to
  `06_reduce/sensitivity_single_block[/stratified_<kernel>]/`, never over the
  D022 primary.
- Every feature entering redundancy clustering must have an ICC entry.

### Single-kernel cohort exclusion (ComBat prerequisite)

Among the 437 eligible patients the kernels are Qr36d/2 (231), I30f/3 (205)
and I36f/3 (1). neuroCombat with a singleton kernel produces NaN because
within-batch variance is undefined. The orchestrator filters to {Qr36d/2,
I30f/3} before matrix prep, so the stage 5 analysis cohort is 436. Defense-in-depth
assertion in `combat_harmonise` raises if any singleton slips through.

## Module contracts

```python
# prepare_matrix.run_matrix_prep
def run_matrix_prep(
    df: pd.DataFrame,              # eligible cohort, pid + metadata + features
    feature_cols: list[str],       # gated features from stage 4
    variance_threshold: float = 0.01,
    yj_skew_threshold: float = 1.0,
    combat_max_post_r2: float = 0.02,
    add_derived: bool = True,
    reliable_transformed: set[str] | None = None,   # D040: 5b derived_icc passing set
) -> tuple[pd.DataFrame, list[str], MatrixPrepLog]:
    """Returns prepared dataframe, list of surviving feature columns, audit log."""

# redundancy.run_multi_block_redundancy_clustering
def run_multi_block_redundancy_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    icc_lookup: dict[str, IccInfo],
    canonical_set: set[str],
    blocks: tuple[FeatureBlock, ...] = DEFAULT_BLOCKS,
    primary_method: LinkageMethod = "average",
    sensitivity_methods: tuple[LinkageMethod, ...] = (),
    min_gap: float = 0.05,
    fallback_distance: float = 0.25,
) -> MultiBlockResult:
    """Returns representatives (union over blocks), per-block clustering, audit."""

# pca.fit_pca
def fit_pca(
    df: pd.DataFrame,
    feature_cols: list[str],       # the representatives
    cumvar_threshold: float = 0.85,
    random_state: int = 42,
) -> PcaResult:
    """Returns components, scores (N x n_retain), pid_order, sign-normalised."""
```

## Decisions

- D017 - sentinel-prone per-vessel feature exclusion at stage 5 ingestion
- D018 - density tier and dense_calcium binarisation
- D019 - preprocessing pipeline (variance filter, ComBat, YJ + rank)
- D035 - ComBat columns derived from the gate (all non-shape PyRadiomics)
- D020 - redundancy clustering and PCA
- D022 - multi-block redundancy clustering (primary; the single-matrix D020 variant is a sensitivity run)
- D040 - vessel Gini over all four vessels; transformed features carry measured D033 reliability

## Tests

`tests/reduce/`:

| Module | Test file |
|---|---|
| prepare_matrix | `test_prepare_matrix.py` |
| redundancy (single + multi-block) | `test_redundancy.py` |
| pca | `test_pca.py` |
| stage 5 script safeguards | `test_stage5_script.py` |

## Results on the analysis cohort (D033 gate)

| Cohort | Patients | Features after D019 | Representatives (D022) | PCs at 85% | Cumulative variance |
|---|---|---|---|---|---|
| Full | 436 | 155 | 35 | 15 | 85.95% |
| Qr36d/2 | 231 | 153 | 33 | 14 | 86.60% |
| I30f/3 | 205 | 154 | 36 | 15 | 86.23% |

- Full cohort: 166 gated features, 13 dropped by D017; both derived features
  admitted by the R^2 < 0.95 gate (`high_density_fraction` R^2 .864,
  `vessel_burden_gini` .918).
- ComBat on 84 PyRadiomics columns: largest kernel R^2 .034 before and
  .0091 after (D019 bound 0.02).
- In the Qr36d/2 stratum both derived features sit within .003 of the 0.95
  cutoff and are rejected, so the strata do not share all of the full
  cohort's derived features.
- The seam file `pca_scores.npy` is byte-exact to the in-memory `pca.scores`.

### Cross-cohort robust feature set (33 features)

Features that pass the D033 gate AND are redundancy-cluster representatives
in the full cohort (35) and in both kernel strata (Qr36d/2 33, I30f/3 36).
Stage 7 writes the list to `outputs/07_analyse/cross_cohort_robust_features.csv`.
Class = D026 relation to `agatston_total` (Spearman rho on raw values, full
cohort). ICC = D033 ICC(1,1); "by construction" marks XML-derived values the
perturbation cannot change (ROI counts per density tier, XML max HU, artery
count), so their reliability is not an empirical result.

| Block | Feature | Class | rho | ICC |
|---|---|---|---|---|
| burden | `mass_total` | burden | 0.998 | 1.000 |
| burden | `mass_lm` | mixed | 0.405 | 0.999 |
| density_tier | `n_rois_d1_rca`, `n_rois_d2_lcx`, `n_rois_d2_rca`, `n_rois_d3_lad`, `n_rois_d3_lcx`, `n_rois_d4_lad`, `n_rois_d4_lcx` | burden | 0.55 to 0.81 | by construction |
| density_tier | `n_rois_d1_lad`, `n_rois_d1_lcx`, `n_rois_d1_lm`, `n_rois_d2_lad`, `n_rois_d2_lm`, `n_rois_d3_lm`, `n_rois_d4_lm` | mixed | 0.09 to 0.49 | by construction |
| density_tier | `has_dense_calcium` | not evaluated in the reported run (absent at raw scale; re-derived from the next run, D026.1) | | by construction |
| hu_statistics | `max_hu_global`, `max_hu_lcx`, `max_hu_rca` | burden | 0.74 to 0.90 | by construction |
| hu_statistics | `max_hu_lm` | mixed | 0.396 | by construction |
| shape | `original_shape_Maximum3DDiameter` | burden | 0.837 | 1.000 |
| shape | `original_shape_Elongation` | **structure** | 0.076 | 0.988 |
| spatial | `n_calcified_arteries` | burden | 0.807 | by construction |
| spatial | `lesion_count_lm` | mixed | 0.365 | 0.998 |
| texture | `original_firstorder_Kurtosis` | burden | 0.725 | 0.983 |
| texture | `original_glcm_Idn` | burden | 0.555 | 0.898 |
| texture | `original_glrlm_RunLengthNonUniformity` | burden | 0.991 | 1.000 |
| texture | `original_glrlm_RunPercentage` | burden | 0.536 | 0.940 |
| texture | `original_glszm_ZoneVariance` | mixed | 0.394 | 0.956 |
| texture | `original_glcm_Correlation` | **structure** | -0.073 | 0.878 |
| texture | `original_ngtdm_Contrast` | **structure** | -0.083 | 0.958 |
| texture | `original_ngtdm_Strength` | **structure** | -0.275 | 0.943 |

By block: density tiers 15, texture 8, HU statistics 4, burden 2, shape 2,
spatial 2. By class (reported run): burden-tracking 17, mixed 11, structure-tracking
4, not evaluated 1 (`has_dense_calcium`, D026.1). The four structure-tracking features (elongation, GLCM
correlation, NGTDM contrast and strength) carry information largely
independent of total burden. 20 features are reliable by construction; the
13 measured ones have ICC 0.878 to 1.000.

## Outputs (seam to discover stage)

Written under `outputs/06_reduce/` (full cohort) or
`outputs/06_reduce/stratified_<kernel>/` (kernel-stratified sensitivity):

| File | Content |
|---|---|
| `pca_scores.npy` + `pca_scores_pid_order.csv` | N x n_retain PC matrix, byte-exact seam |
| `pca_scores.csv` | same data as the NPY, lossy CSV form for human inspection |
| `prepared_matrix.csv` | post-D019 features (pid + metadata + 77 transformed) |
| `cohort_metadata.csv` | RAW kernel, agatston_total, low_burden_flag, category |
| `representative_features.csv` | 29 multi-block representatives (full cohort) |
| `multi_block_assignments.csv` | per-feature: block, cluster id, is_representative, decided_by |
| `multi_block_summary.json` | per-block cluster count + cut threshold + reps |
| `pca_explained_variance.csv` | per-PC variance and cumulative variance |
| `pca_top_loadings.csv` | top 10 absolute loadings per PC with family tag |
| `pca_loadings.csv` | full per-feature loadings |
| `pc_agatston_correlation.csv` | Spearman rho between each PC and agatston_total |
| `combat_audit.csv` | pre / post kernel R^2 on each ComBat target |
| `matrix_prep_log.json` | D019 step-by-step audit |
| `run_header.json` | git hash, library versions, CLI args, timestamp |

## How to run

```bash
# Production full cohort (~1 min)
python scripts/06_reduce.py --n-jobs 80

# D022 sensitivity probe (single-matrix Spearman); writes to
# outputs/06_reduce/sensitivity_single_block/
python scripts/06_reduce.py --block-mode single --n-jobs 80

# Kernel-stratified reruns (D021 sensitivity cohorts)
python scripts/06_reduce.py --kernel-filter "Qr36d/2" --n-jobs 80
python scripts/06_reduce.py --kernel-filter "I30f/3" --n-jobs 80
```

Each command writes the seam files for the downstream `07_discover.py`.

## Known limitations

- ComBat fits on 2 kernels with N=220 / 200. Convergence is fine but
  N=200 is on the lower end of ComBat's well-tested regime. The post-correction
  R^2 < 0.0025 confirms it worked, but the theoretical justification weakens
  on smaller batches.
- PCA at 0.85 cumvar gives n_retain = 13 on the full cohort, but this varies
  by stratum. Downstream comparisons across cohorts must be on PC scores, not
  on PC component identities.
- The composition argument for multi-block (D022) is prospective but somewhat
  judgement-based. Sensitivity vs `--block-mode single` is the empirical
  defence.
