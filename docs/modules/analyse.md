# Stage 7, analyse (phenotype characterisation)

## What this stage does

Takes the partitions produced by stage 6 (spatial-only k=2 and forced k=3
in the PC space, plus an agatston tertile partition computed at stage 7
entry) and characterises each phenotype cluster with publication-grade
statistical rigour. It does NOT re-cluster, NOT re-select features, and
NOT validate externally; that is the job of stage 6 and stage 8.

The analysis runs in 13 ordered steps in the orchestrator
(`scripts/08_analyse.py`):

  0. Augment raw features with the two D019 derived features
     (`high_density_fraction`, `vessel_burden_gini`) using the
     byte-identical formulas from `predict.reduce.prepare_matrix`.
  1. Label balance check on every spatial_k2 partition (D023).
  2. Focal / diffuse mapping (D023): lower median `n_calcified_arteries`
     = focal.
  3. Biological sanity check (D023): focal median `max_hu_global` must
     be >= 130 HU (the IBSI / Agatston calcium definition). Fail loud
     on a soft-tissue-mask regression.
  4. Per-cohort cluster profiles (D023): median + IQR + Cliff's delta +
     Mann-Whitney + FDR-BH across the feature bundle (the cross-cohort
     robust representatives + 13 spatial PCA inputs, deduplicated).
  5. Top-N signatures per (cohort x partition x cluster).
  6. Burden orthogonality (D024): Mann-Whitney + Levene + Cliff's delta
     on `agatston_total` between focal and diffuse, with 3-level
     `interpretation` column (orthogonal / marginal / confounded).
  7. Burden-stratified spatial replication (D024 part 2): within each
     Agatston tertile, re-check the 6 directional hypotheses.
  8. Directional hypothesis test (D025): 6 pre-registered one-sided
     Mann-Whitney tests + FDR-BH on the family + two-tier verdict
     (primary 4/6 confirmed; secondary 4/6 direction-match in both strata).
  9. Monotonicity classification (D026): Spearman + Kendall vs
     `agatston_total` for each cross-cohort robust feature; classify into
     burden_tracking / structure_tracking / spatial_tracking / mixed; a
     feature with no computable rho is not_evaluable (D026.1).
 10. Cross-cohort feature consistency (D027): three-rule criterion
     (direction consistent + significance in >= 2 of 3 + |delta| >= 0.20
     in all 3).
 11. Partition ARI on shared pids (D027 complementary check).
 12. Main paper table (D028: 15 rows = 3 cohorts x 5 phenotype clusters).
 13. Low-burden sensitivity (D028, D028.1): the low_burden_flag = False
     subset of the full cohort, full-cohort spatial labels kept: D024,
     D025 primary rule, profiles and signatures, the 5-row table, and a
     verdict stating whether each result matches the full cohort.
 14. `run_header_analyse.json` with SHA of every seam file consumed.

Before step 0 every cohort's stage 6 record is verified, and features.csv
must be byte-identical to the file stage 5 read (hash in
`COMPLETE_stage5.json`). Outputs go to `outputs/07_analyse.partial/` and are
published with `COMPLETE_stage7.json` only after a complete run. The report
(08d), the focal/diffuse verification (08e), stage 8 and the figures call
`predict.analyse.stage7.require_stage7_complete`.

## Module list

| Module | Concern |
|---|---|
| `predict/analyse/derived_features.py` | Re-derive D019 features (`high_density_fraction`, `vessel_burden_gini`) at raw clinical scale; byte-identical to `predict.reduce.prepare_matrix.compute_*`. |
| `predict/analyse/profiles.py` | Cliff's delta + Mann-Whitney + FDR-BH; biological sanity (130 HU floor); label balance (>15% minority); focal/diffuse mapping rule. |
| `predict/analyse/orthogonality.py` | Burden orthogonality (Mann-Whitney + Levene + Cliff's delta on agatston) with 3-level interpretation; burden-stratified spatial replication. |
| `predict/analyse/hypotheses.py` | Pre-registered 6 directional hypotheses; one-sided Mann-Whitney; primary + secondary + overall verdict. |
| `predict/analyse/monotonicity.py` | Spearman + Kendall vs agatston; burden / structure / spatial / mixed classification. |
| `predict/analyse/signatures.py` | Top-N ranking by FDR-adjusted Cliff's delta per (cohort x partition x cluster). |
| `predict/analyse/cross_cohort.py` | 3-rule feature-level consistency + partition ARI on shared pids. |
| `predict/analyse/paper_table.py` | 15-row main paper table + 5-row robust sensitivity; Agatston categories 0, (0, 100), [100, 400), >= 400 (exhaustive for fractional scores). |
| `predict/analyse/sensitivity.py` | Low-burden sensitivity (D028, D028.1): D024, D025 primary rule, signatures, verdict. |
| `predict/analyse/stage7.py` | Stage 7 inputs check (stage 6 records, features.csv vs stage 5) and `COMPLETE_stage7.json` verification. |
| `scripts/08_analyse.py` | Orchestrator. |
| `scripts/08d_stage7_results.py` | Read-only presentation report; every number and verdict read from the current files (no fixed text). |
| `scripts/08e_verify_focal_diffuse.py` | Independent re-derivation harness for the focal/diffuse chain. |

## Why these design choices

### D023: Cliff's delta over Cohen's d

Cliff's delta is rank-based, dimensionless, and robust to outliers and
skew. Cohen's d on z-scored data is the radiomics-publication convention
but assumes Gaussian inputs. On 93%-sparse LM density-tier bins, Cohen's
d explodes to physically meaningless values, and on heavy-tailed Agatston
scores it under-weights the long tail. Romano 2006 thresholds: 0.147
(small), 0.330 (medium), 0.474 (large). We adopt |delta| >= 0.20 as the
robust-discriminator floor.

### D023: biological sanity = absolute 130 HU floor (not a relative ratio)

A relative rule such as `focal max_hu >= 0.9 * diffuse max_hu` fires on the
real cohort (the COCA ratio is about 0.5; focal calcium is legitimately
lower-peak than diffuse). The correct gate is the IBSI
calcium definition: focal cluster median max_hu_global must be >= 130 HU
(the Agatston calcium threshold). A pipeline regression that puts
soft-tissue voxels in the focal cluster would push max_hu to near-zero,
far below 130. The ratio is logged as a WARNING when below 0.5, not
raised.

### D024: 3-level interpretation column on burden orthogonality

A binary pass/fail collapses two qualitatively different scientific
situations: (a) p < 0.05 with |delta| < 0.20 (significant but trivial
effect in a large cohort) is a PASS, but the trivial-effect nature must
be visible; (b) p >= 0.05 with |delta| >= 0.20 (underpowered with
visible effect in a small cohort) is a PASS, but the underpowered-but-
visible nature must be visible. The orthogonal / marginal / confounded
3-level column lets the paper sentence read "orthogonal in cohort A,
marginal in cohort B, confounded in cohort C" cleanly.

### D025: pre-registered 6 directional hypotheses, two-tier verdict

The 6 hypotheses are LOCKED. They were specified in D025 before any
stage 7 numerical result was observed. Primary requires >= 4 of 6
confirmed at FDR p < 0.05 in the full cohort. Secondary requires >= 4
of 6 direction-match in BOTH strata (direction only; significance not
required). The secondary check protects against full-cohort pass driven
by kernel imbalance.

### D026: classification rule uses both rho AND block

Spearman |rho| >= 0.5 -> burden_tracking regardless of block (the
correlation strength dominates). For |rho| < 0.3, the block tag becomes
the tiebreaker (spatial block -> spatial_tracking; hu / texture /
shape -> structure_tracking; otherwise mixed). The intermediate band
0.3 <= |rho| < 0.5 is always mixed. This prevents the procedure from
re-classifying a structurally spatial feature as burden-tracking on
weak-rho evidence.

### D027: three-rule cross-cohort criterion + complementary ARI

ARI alone tests partition identity (same patient -> same cluster across
cohort runs). Feature-level rules test feature discriminator identity
(same feature distinguishes the same way across cohorts). Both belong
in the paper as separate evidence pillars. The three rules together
form a stable replication-grade discriminator definition: direction
consistent (rule 1), significant in at least 2/3 cohorts (rule 2),
moderate effect in ALL 3 cohorts (rule 3, the safety net against
effect-size dilution).

### D028: robust cohort sensitivity as a SEPARATE file

Adding the robust cohort as a 4th row block in the main paper table
would conflate cross-kernel replication (D027) with low-burden
sensitivity (D028). The separate file lets the paper's Methods section
reference it explicitly: "as a sensitivity analysis, we re-ran the
characterisation on the patients with low_burden_flag = False
(Supplementary Table X)".

D028.1 adds the orthogonality and directional-hypothesis files for the
subset and a verdict rule: the same D024 interpretation and D025 primary rule (>= 4 of
6 confirmed) on the restricted patients, each reported as the same as or
different from the full cohort. No Hennig value is reported for the subset
(stage 6 Hennig describes full-cohort clusters).

## Module contracts

```python
# profiles.compute_cluster_profile
def compute_cluster_profile(
    raw_features: pd.DataFrame,
    labels: pd.Series,
    feature_names: list[str],
    cohort: str,
    partition: str,
) -> pd.DataFrame:
    """One row per (cluster, feature). Insufficient_data excluded from FDR-BH."""

# profiles.apply_fdr_bh
def apply_fdr_bh(
    profile_df: pd.DataFrame,
    p_column: str = "mannwhitney_u_pval",
    out_column: str = "fdr_bh_pval",
    delta_threshold: float = 0.20,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """BH adjustment within each (cohort, partition) bundle; adds
    is_robust_discriminator column."""

# orthogonality.assess_burden_orthogonality
def assess_burden_orthogonality(
    agatston_focal: np.ndarray, agatston_diffuse: np.ndarray, cohort: str,
) -> BurdenOrthogonalityResult:
    """Mann-Whitney + Levene + Cliff's delta + 3-level interpretation."""

# hypotheses.directional_hypotheses_table
def directional_hypotheses_table(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    cohort: str,
    hypotheses: tuple[tuple[str, Direction], ...] = DIRECTIONAL_HYPOTHESES,
) -> pd.DataFrame:
    """One row per hypothesis with one-sided MW + FDR-BH + confirmed."""

# monotonicity.compute_monotonicity
def compute_monotonicity(
    raw_features: pd.DataFrame, feature_names: Iterable[str],
    agatston: pd.Series, cohort: str, block_lookup: dict[str, str],
) -> pd.DataFrame:
    """Spearman + Kendall + 4-class classification per feature."""

# cross_cohort.consistency_table
def consistency_table(
    profiles_by_cohort: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """3-rule criterion per (feature, partition, cluster) tuple."""

# derived_features.augment_raw_with_derived
def augment_raw_with_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add high_density_fraction + vessel_burden_gini columns (byte-
    identical to predict.reduce.prepare_matrix.compute_*) if their input
    columns are present."""
```

## Decisions

- D023 - per-cluster characterisation (Cliff's delta + FDR-BH; 130 HU
  absolute biological sanity floor; 15% minority label balance)
- D024 - burden orthogonality protocol (Mann-Whitney + Levene + delta;
  3-level interpretation; burden-stratified replication)
- D025 - directional hypothesis test (6 pre-registered + two-tier verdict)
- D026 - monotonicity classification (Spearman / Kendall; 4-class rule)
- D026.1 - no computable rho is `not_evaluable`; `has_dense_calcium` re-derived at raw scale
- D027 - cross-cohort feature consistency (3-rule + ARI complementary)
- D028 - low-burden sensitivity (separate output file)

## Tests

`tests/analyse/`:

| Module | Covers |
|---|---|
| derived_features | byte-identity vs stage 5 |
| profiles | Cliff + Mann-Whitney + FDR + sanity + balance + focal map |
| orthogonality | 3-level interpretation + Levene + tertile replication |
| hypotheses | 6 directional + primary + secondary + overall |
| monotonicity | Spearman + Kendall + classification + edge cases |
| signatures | top-N + alphabetical tiebreak + paragraph formatter |
| cross_cohort | 3-rule + ARI on shared pids |
| paper_table | 15 rows + sensitivity + Hennig lookup + tertile balance |
| stage 7 script | record checks, raw-feature binding, report from current results |

## Results on the analysis cohort

Spatial-only GMM k=2 partition from stage 6 (full 436, Qr36d/2 231,
I30f/3 205). Every number below is in `outputs/07_analyse/`.

### Burden orthogonality (D024)

| Cohort | focal median Agatston | diffuse median Agatston | Cliff's delta | interpretation |
|---|---|---|---|---|
| full | 26 | 419 | -0.900 | confounded |
| Qr36d/2 | 34 | 630 | -0.913 | confounded |
| I30f/3 | 20 | 318 | -0.894 | confounded |

The two spatial clusters are **not orthogonal to total calcium burden** in
any cohort.

### Directional hypotheses (D025)

Primary rule met: 4 of 6 confirmed in the full cohort; 4 of 6 direction
matches in each kernel stratum. **Verdict: robust.**

- Confirmed: `n_calcified_arteries` (focal fewer), `dist_from_top_max`,
  `vessel_burden_gini` (focal median 0.75, the value for all burden in one
  vessel), `first_to_last_dist_lad` (focal median 0).
- Refuted and reversed: `lesion_count_lad` (focal 1 vs diffuse 4 to 5) and
  `gini_lesion_volume` (focal median 0.08 / 0.19 / 0.00).

The confirmations follow mechanically from focal patients having one
calcified vessel and one or two lesions (one vessel gives vessel Gini 0.75;
one lesion gives lesion-volume Gini 0 and LAD extent 0). The two
concentration predictions that do not follow from counts are reversed.

### Other outputs

- Low-burden sensitivity (D028.1, 341 patients): D024 confounded (Cliff's
  delta -0.819) and D025 primary 4 of 6, both the same as the full cohort.
- D027: 150 of 215 feature x partition x cluster comparisons meet the
  three-rule cross-cohort criterion, dominated by burden-driven features.
- D026 (33 cross-cohort robust features, full cohort): 17 burden-tracking,
  11 mixed, 4 structure-tracking, 0 spatial-tracking; `has_dense_calcium`
  was not evaluated in this run (counted as mixed by the code at the time;
  D026.1).
- Cross-cohort ARI on shared patients (D027 / D031, threshold 0.80):
  spatial k=2 Qr36d/2 .787 (below), I30f/3 .813; burden tertiles k=3
  Qr36d/2 .807, I30f/3 .785 (below).

## Interpretation (Finding 3)

The Hennig-stable spatial k=2 partition (median Jaccard 0.84 to 0.94
across cohorts) is reproducible but is best described as a
**low-burden versus high-burden dichotomy** projected through the spatial
subspace, not a burden-independent focal-versus-distributed phenotype:

1. Burden orthogonality is confounded in all three cohorts (Cliff's delta
   about -0.9).
2. D025 passes its pre-registered rule, but its confirmations are
   mechanical consequences of focal patients having one calcified vessel
   and one or two lesions, while the concentration predictions are
   reversed. Passing D025 does not establish burden independence, so D024
   and D025 are reported together.
3. Focal patients have a median of one calcified artery and Agatston about
   20 to 35; diffuse patients have three arteries and Agatston about 300 to
   650.

## Outputs (seam to stage 8)

Written under `outputs/07_analyse/`:

| File | Content |
|---|---|
| `cluster_profiles.csv` | per (cohort x partition x cluster x feature): median, IQR, Cliff's delta, MW p, FDR-BH p, is_robust_discriminator |
| `signature_features.csv` | top-N per cluster from FDR-adjusted Cliff's delta |
| `burden_orthogonality.csv` | per cohort: MW p, Levene p, Cliff's delta on agatston, 3-level interpretation, passes |
| `burden_stratified_spatial.csv` | per (cohort x tertile x feature): direction match within tertile |
| `directional_hypotheses.csv` | per (cohort x hypothesis): focal/diffuse median, observed sign, one-sided MW p, FDR-BH p, confirmed |
| `directional_verdict.json` | primary + secondary + overall verdict |
| `cross_cohort_robust_features.csv` | the cross-cohort robust feature set: gated, non-redundant representatives in the full cohort and in both kernel strata |
| `monotonicity_classification.csv` | per (cohort x feature): Spearman, Kendall, classification |
| `monotonicity_summary.csv` | per cohort: count per classification |
| `cross_cohort_feature_consistency.csv` | per (feature x partition x cluster): 3-rule + robust_discriminator |
| `cross_cohort_robust_counts.csv` | aggregated robust discriminator counts |
| `cross_cohort_ari.csv` | partition ARI on shared pids per (partition x stratum) |
| `phenotype_paper_table.csv` | 15-row main paper table |
| `phenotype_paper_table_robust_sensitivity.csv` | 5-row robust sensitivity table (signatures recomputed on the subset) |
| `burden_orthogonality_robust_sensitivity.csv` | D024 on the low-burden-excluded subset |
| `directional_hypotheses_robust_sensitivity.csv` | D025 hypotheses on the subset |
| `cluster_profiles_robust_sensitivity.csv`, `signature_features_robust_sensitivity.csv` | profiles and signatures on the subset |
| `robust_sensitivity_verdict.json` | D028.1 verdict: subset vs full cohort for D024 and the D025 primary rule |
| `run_header_analyse.json` | git hash + library versions + CLI args + seam SHA per cohort + biological sanity per cohort + verdict + inputs |
| `COMPLETE_stage7.json` | completion record: inputs (stage 6 record hashes, features.csv hash), settings, code fingerprint, output hashes |

## How to run

```bash
# Production stage 7 across all 3 cohorts
python scripts/08_analyse.py

# Read-only presentation report
python scripts/08d_stage7_results.py

# Independent verification of the focal/diffuse chain (debugging)
python scripts/08e_verify_focal_diffuse.py
```

Wall-clock at production scale: ~1 second total. Stage 7 has no
clustering, no bootstrapping, no parallelism.

## Known limitations

- The orchestrator reads RAW values from `features.csv`; the two D019
  derived features (`high_density_fraction`, `vessel_burden_gini`) are
  re-derived at raw scale via `augment_raw_with_derived` because they
  do not exist in `features.csv`. Byte-identity to the stage-5 formulas
  is enforced by `tests/analyse/test_derived_features.py`.
- Forced k=3 burden tertiles are computed at stage 7 entry via
  `pd.qcut(agatston_total, q=3)`, NOT taken from the stage-6 forced k=3
  PC-space partition. The qcut tertiles are the conservative,
  distribution-defined choice; the PC-space forced k=3 is reported by
  stage 6 only for ARI cross-algorithm comparison.
- The 13 spatial features used as RAW inputs to the spatial PCA differ
  from the cross-cohort representatives; the analysis bundle is their
  union. A PyRadiomics feature can fall out of the per-cluster profile when
  it is NaN for the patients whose mask is below PyRadiomics' minimum ROI
  size (D010).
- Stage 7 is descriptive, not predictive. No MACE association, no
  demographic adjustment (demographics absent in COCA), no external
  cohort validation. Those are stage 8 (or beyond) concerns.
