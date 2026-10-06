# Pipeline walk-through

`run_pipeline.sh` runs stages 1 to 10 in order (`--from N --to M` for a
range; stage 11 self-checks run only on request). Per-stage detail
(decisions, results, edge cases) is in `docs/modules/<stage>.md`.

Every stage writes to `<folder>.partial/` and publishes the folder in one
step together with a completion record (input hashes, code fingerprint,
output hashes). A consumer verifies that record before reading
(`require_stageN_complete`), so a partial, stale or smoke result is never
used. Logs go to `outputs/_logs/`.

| Stage | Package / scripts | Output folder |
|---|---|---|
| 1 discovery | `io` · `01_discover.py` | `outputs/01_manifest/` |
| 2 preprocess | `preprocess` · `02_preprocess.py` | `outputs/02_preprocessed/` |
| 3 features | `features` · `03_features.py`, `03b_cohort_sanity.py` | `outputs/03_features/` |
| 4 stability | `stability` · `04_*`, `05_*`, `calibrate_realistic.py` | `outputs/04_perturbations/`, `04b_realistic/`, `05_icc/`, `05b_icc_realistic/` |
| 5 reduce | `reduce` · `06_reduce.py` | `outputs/06_reduce/` (+ `stratified_<kernel>/`) |
| 6 discover | `discover` · `07_discover.py` | written into the stage 5 cohort folders |
| 7 analyse | `analyse` · `08_analyse.py`, `08d_stage7_results.py`, `08e_verify_focal_diffuse.py` | `outputs/07_analyse/` |
| 8 validate | `validate` · `09_validate.py` | `outputs/08_validate/` |
| 9 exploratory | `experiments/*` · `09_stage9.py begin / finish` | `outputs/exploratory/` |
| 10 figures | `figures` · `10_figures.py` | `outputs/figures/` |
| 11 self-checks | `diagnose_kernel_leak.py`, `diagnose_kernel_chi2.py`, `probe_detectability_floor.py` | `outputs/checks/`, `outputs/detectability_d0361/` |

## 1. `io`: discover and load

- `io.patient_discovery`: intersect `data/raw/` and `data/calcium_xml/`,
  select one series per patient, check slice geometry, apply exclusions
  (D004, D039, D039.1); writes `manifest.csv` and `exclusions.csv`.
- `io.dicom_loader`: load the selected series, Z-sorted (D006).
- `io.xml_parser`: parse the Apple plist annotations, keeping `Center` for
  Z matching (D001) and the XML HU statistics.
- `io.spacing`: spacing metadata I/O.

## 2. `preprocess`: mask and resample

- `preprocess.slice_matcher`: ROI to slice by Z coordinate (D001).
- `validate.xml_roundtrip`: Max-exact XML round-trip; failing ROIs are
  excluded from the mask (D002, D012).
- `preprocess.mask_builder`: XML polygons to a 3D binary mask.
- `preprocess.resampling`: CT and mask to 0.375 x 0.375 x 3.0 mm (D005).
- `preprocess.hu_handling`: HU clip and metal flag (D003).

## 3. `features`: extract

- `features.agatston`, `per_vessel_aggregates`, `density_tiers`: XML-derived
  burden, mass, density features (D011, D018).
- `features.lesion_ccl`: 3D lesions by BFS grouping (D007).
- `features.spatial`: lesion count, distances, diffusivity, Gini.
- `features.radiomics`: PyRadiomics, all 7 IBSI families (D009, D010).
- `features.per_artery_mask`: per-vessel masks for the sanity audit (D008).

## 4. `stability`: perturb and gate

- `stability.realistic`: D033 composite draws and calibration (primary gate,
  ICC(1,1)).
- `stability.perturbations`: D014 fixed-mask perturbations (supplementary,
  ICC(3,1)).
- `stability.icc`, `stability.tiers`: ICC formulas, gate, D034 tiers.

## 5. `reduce`: prepare, de-duplicate, project

- `reduce.prepare_matrix`: D017 drops, derived features, variance filter,
  ComBat (D019, D035, D040), Yeo-Johnson with rank fallback, z-score.
- `reduce.redundancy`: multi-block Spearman redundancy clustering and
  representative selection (D020, D022).
- `reduce.pca`: PCA at 85% cumulative variance, sign-normalised.

## 6. `discover`: clusterability and validity

- `discover.clusterability`: Hopkins statistic (D021.1).
- `discover.cluster_discovery`: kmeans / ward / GMM, gap statistic,
  consensus clustering, burden residualisation (D021).
- `discover.validity`: kernel association with burden adjustment, Hennig
  bootstrap stability, ARI (D021.1).
- `discover.detectability`, `discover.taxometric`: method checks (D036,
  D038).

## 7. `analyse`: characterise

Per-cluster profiles (D023), burden orthogonality (D024), pre-registered
directional hypotheses (D025), monotonicity (D026), cross-cohort feature
consistency (D027), low-burden sensitivity (D028, D028.1).

## 8. `validate`: reproduce and transfer

External GE holdout (D029), leave-k-out cross-validation (D030),
cross-cohort ARI (D031), with production-fit parity (D041).

## 9. Exploratory lesion analyses

Pre-registered analyses in `experiments/` (lesion morphology, LAD phenotype,
isolated left-main disease), run as one published chain (D043).

## 10. Figures

Paper figures built from the published stage 6, 7 and 9 outputs (D044).
