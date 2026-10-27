# Radiomics Feature Extraction and Calcium Phenotype Discovery

A reproducible framework that extracts radiomics features from coronary artery calcium (CAC)
masks, selects the features that survive realistic acquisition and annotation variation,
discovers calcium phenotypes with unsupervised clustering, and validates them **without
clinical endpoints**. Built for **Google Summer of Code 2026 (ML4SCI)** on the **Stanford
COCA** dataset of gated, non-contrast cardiac CT with expert calcium annotations.

The Agatston score summarises calcium as one number. This project asks what the shape,
texture, density and spatial distribution of calcium add to that number, and builds the
tools to answer it reproducibly.

## Highlights

- **Complete feature pipeline on every usable gated COCA scan.** 175 features per patient
  (68 XML-derived burden, density, per-vessel and spatial features plus 107 PyRadiomics
  shape, first-order and texture features) for 439 scans, plus 3,171 individual 3D lesions.
- **A robust feature set.** 166 of 175 features pass a realistic perturbation gate that
  combines positioning, grid shift, field of view, reconstruction kernel, dose noise, HU
  calibration drift and annotator-boundary jitter (ICC >= 0.75 over 20 draws per patient).
- **Distinct lesion-level phenotypes.** Lesions cluster strongly (Hopkins 0.95; 12 clusters,
  all bootstrap-stable), group into soft, moderate and dense calcium, and include a dense,
  RCA-dominant plaque type with zero left-main lesions that is recovered at every cluster
  count from 7 to 12, in both scanner-kernel subgroups and under annotation perturbation.
- **Anatomical patterns.** Lesion types are tied to vessel territories (RCA, LAD, LCx), and
  a small low-burden group with isolated left-main calcification stands out.
- **An endpoint-free validation framework** with Agatston stratification, cluster-quality
  and bootstrap stability, perturbation reproducibility, cross-scanner replication, an
  external-scanner holdout, clinical-pattern alignment, and planted-cluster simulations
  that check whether the discovery methods can find structure when it exists.

## Project goals and what was delivered

| ML4SCI goal | Delivered |
|---|---|
| Shape, intensity, texture and spatial features with PyRadiomics and SimpleITK | 107 PyRadiomics features (all 7 IBSI families) and 68 canonical features (Agatston, volume, mass, density tiers, per-vessel burden, lesion count, inter-lesion distances, diffusivity, Gini concentration) for all 439 scans ([features](docs/modules/features.md)) |
| A robust, documented pipeline with 50 to 100 key features | 166 features pass the realistic reliability gate; redundancy clustering keeps 35 representative features for the patient analysis, 33 of them in all three cohorts (the key robust set); each of 10 stages documented in `docs/modules/` and backed by a dated decision record |
| PCA "eigen features" | 15 principal components explain 86% of the variance of the representative features ([reduce](docs/modules/reduce.md)) |
| Unsupervised clustering (K-means, hierarchical, GMM, DBSCAN) | K-means, Ward hierarchical and GMM with gap statistic, consensus clustering and Hennig bootstrap stability, on three feature spaces (full, burden-removed, spatial-only) and on lesions ([discover](docs/modules/discover.md)) |
| Feature-space visualisation and Agatston correlation | PCA projections by Agatston tertile (Figure 1); burden orthogonality tests and per-feature burden monotonicity ([analyse](docs/modules/analyse.md)) |
| Perturbation invariance (rotation ±5°, translation, noise) | Exactly that set (D014: rotations ±5° and ±10°, translations ±2 and ±5 mm, noise 5 and 10 HU) plus a realistic composite set anchored to COCA's own variation (D033), with per-arm results ([stability](docs/modules/stability.md)) |
| Literature patterns (spotty / dense / diffuse) | Lesions grouped into soft-microspot, moderate and dense classes by fixed HU / volume rules; patients split into focal and diffuse disease; both compared with the unsupervised clusters |
| Vessel territory distribution | Per-cluster vessel enrichment (obs/exp, chi-square, Cramer's V); RCA-, LAD- and LCx-biased lesion clusters |
| Main phenotype distinguishers | Effect-size ranking (Cliff's delta with FDR control) replicated across scanner subgroups, plus burden / structure / spatial classification of each feature |
| Validation without clinical endpoints | Agatston stratification, cluster quality, reproducibility, clinical alignment, cross-scanner replication, external GE holdout, leave-k-out cross-validation, planted-cluster detectability checks |

## Key findings

### 1. A reliable, interpretable feature set

Under the realistic perturbation gate, all 68 canonical features and 98 of 107 PyRadiomics
features are reliable (166 / 175). The 9 that fail are low-intensity and zone-size texture
descriptors. The simulated variation stays below published rescan variability (Agatston
4.6% between draws vs about 20% between real rescans), so the gate tests realistic, not
extreme, conditions. The classic fixed-mask set (rotation, translation, noise) keeps 89
features; every one of them also passes the realistic gate. After redundancy clustering,
35 representative features, selected within six pre-defined blocks (burden, HU statistics,
density tiers, spatial, texture, shape), carry the patient analysis.

**The key robust feature set: 33 features** that are reliable and non-redundant in the full
cohort and in both scanner-kernel subgroups: 15 per-vessel density-tier counts, 8 texture
features, 4 maximum-HU values, total and left-main mass, two shape features (maximum 3D
diameter, elongation), the number of calcified arteries and the left-main lesion count.
Seventeen track burden, and four carry information largely independent of it: shape
elongation, GLCM correlation, and NGTDM contrast and strength. Twenty of the 33 are
XML-derived values that the perturbations cannot change, so their reliability holds by
construction; the 13 measured features have ICC 0.88 to 1.00. Full table:
[docs/modules/reduce.md](docs/modules/reduce.md#cross-cohort-robust-feature-set-33-features).

### 2. Patient level: burden is the dominant axis

- The most reproducible patient split separates **focal** disease (median one calcified
  artery, Agatston about 26) from **diffuse** disease (three arteries, Agatston about 420).
  It is stable under bootstrap (Hennig Jaccard 0.84 to 0.94) within the full cohort and
  both scanner subgroups, reproduces under leave-k-out refitting (median ARI 0.82), and transfers to
  four held-out GE scans (assigned by their burden).
- Agatston correlation shows that this split follows total burden (Cliff's delta -0.90
  between the groups). It matches the clinical focal / diffuse
  pattern, and it is best read as a burden gradient, not as a burden-independent subtype.
- The planted-cluster simulations show why this matters: on a strongly skewed burden axis
  the gap statistic and taxometric tests cannot tell groups from a continuum, so the
  framework reports them descriptively and claims neither.
- Scanner kernel is not a hidden driver: the kernel-by-cluster association disappears once
  burden is accounted for.

This is a useful result in its own right: any morphology-beyond-Agatston claim has to be
tested against burden, and the framework does that by construction.

### 3. Lesion level: clear phenotypes with anatomical signatures

- **Strong lesion structure**: 3,171 lesions, Hopkins 0.95, 12 clusters that are all
  bootstrap-stable, organised into 1 soft, 4 moderate and 7 dense clusters.
- **A robust dense RCA plaque type (C8)**: large (median 290 mm³), dense (max HU 842),
  multi-slice lesions, RCA-enriched 2.6-fold with zero left-main lesions; found at every
  k from 7 to 12, in both scanner subgroups, and 94% stable under annotation perturbation.
  Its carriers have extensive disease elsewhere (other-lesion volume 990 vs 79 mm³).
- **An LAD-dominant dense type**: two clusters over-represented in the LAD whose lesions sit
  higher in the vessel than other LAD lesions (p < 0.001), replicated in both subgroups.
- **Isolated left-main calcification at low burden**: in the lowest Agatston tertile, 11
  patients stand apart in feature space and 9 of them have left-main calcium, against 16
  of the other 135 (p = 1e-6). Their lesions are dense, not soft microspots.
- **Composition tracks burden**: dense lesion types make up a larger share of disease at
  higher Agatston tertiles (9 of 12 clusters, Jonckheere-Terpstra p < 0.001).

Whether these lesion types carry information beyond total burden could not yet be settled:
too few burden-matched pairs were available in this cohort. Details and every number:
[lesion morphology](experiments/lesion_morphology/findings.md),
[LAD phenotype](experiments/lad_phenotype/findings.md),
[isolated LM](experiments/lm_isolated_low_burden/findings.md).

## Validation framework (no clinical endpoints)

| Axis | Methods |
|---|---|
| Agatston stratification | burden orthogonality (Mann-Whitney, Levene, Cliff's delta), burden tertiles, feature monotonicity, burden-matched comparisons |
| Cluster quality | Hopkins statistic, gap statistic, consensus clustering (PAC), Hennig bootstrap Jaccard |
| Reproducibility | ICC gates under two perturbation sets, lesion-cluster stability under annotation jitter, leave-k-out cross-validation |
| Generalisation | replication within each scanner-kernel subgroup, external GE-scanner holdout |
| Clinical alignment | soft / moderate / dense lesion classes, focal / diffuse disease, vessel-territory enrichment |
| Method checks | planted-cluster detectability and taxometric simulations; every threshold fixed in `decisions/` before the result it governs |

## Pipeline

```
io -> preprocess -> features -> stability -> reduce -> discover -> analyse -> validate
   -> lesion analyses -> figures
```

| Stage | Purpose |
|---|---|
| 1 `io` | Patient discovery, DICOM series selection, slice-geometry checks, calcium-XML parsing |
| 2 `preprocess` | ROI-to-slice matching by Z coordinate, 3D mask rasterisation, resampling to 0.375 x 0.375 x 3.0 mm, XML round-trip check |
| 3 `features` | Agatston scoring, 3D lesion grouping, per-vessel and spatial features, density tiers, PyRadiomics |
| 4 `stability` | Realistic (primary) and fixed-mask (supplementary) perturbation gates; robust feature set per arm and final |
| 5 `reduce` | Variance filter, ComBat kernel harmonisation, Spearman redundancy clustering, PCA |
| 6 `discover` | Hopkins, gap statistic, consensus clustering, Hennig stability, kernel association |
| 7 `analyse` | Cluster characterisation, burden orthogonality, pre-registered hypotheses, cross-cohort consistency |
| 8 `validate` | External GE holdout, leave-k-out cross-validation, cross-cohort ARI |
| 9 lesions | Pre-registered lesion-level analyses in `experiments/` |
| 10 figures | Six main and five supplementary figures from the published outputs |

Every stage writes to a temporary folder and publishes it in one step with a completion
record (input hashes, code fingerprint, output hashes). Each consumer verifies that record,
so a partial or stale result is never used.

## Repository layout

```
configs/      YAML configuration (paths, seeds, thresholds, perturbations)
params.yaml   PyRadiomics extraction settings
src/predict/  Library code, one subpackage per stage
scripts/      Stage scripts (numbered) and the diagnostic / probe scripts cited by the decisions
tests/        pytest suite mirroring src/predict/
experiments/  Pre-registered lesion-level analyses, each with a plan and findings
decisions/    Design-decision registry, one dated record per decision
docs/         Per-stage documentation, cohort and dataset reports, reproducibility notes
```

## Data

This repository contains **code only**. The Stanford COCA dataset is available from the
Stanford AIMI shared datasets and is not redistributed here. Place the DICOM series and
calcium XML annotations at the paths in `configs/default.yaml` (`data/raw/`,
`data/calcium_xml/`). Cohort details and exclusions: [docs/data.md](docs/data.md).

## Getting started

```bash
# Python 3.8, conda recommended
conda install -c radiomics -c conda-forge pyradiomics
pip install -r requirements.txt
pip install -e .

pytest tests/                                  # unit and integration tests
pytest experiments --import-mode=importlib     # experiment tests
bash run_pipeline.sh                           # stages 1-10 (requires the dataset)
bash run_pipeline.sh --from 6                  # rerun from stage 6
```

Worker count comes from `PREDICT_N_JOBS` or the scheduler's CPU allocation (`PBS_NCPUS`,
`NCPUS`, `SLURM_CPUS_ON_NODE`); `submit_pipeline.sh` is a PBS Pro wrapper. Every stochastic
step is seeded, and parallel results do not depend on the number of workers.

## Limitations and next steps

- COCA has no outcome data, so no result here is validated against MACE. The framework is
  ready to test the lesion phenotypes and the isolated left-main group on a cohort with
  follow-up.
- Burden-matched tests of the lesion types need a larger cohort.
- Not yet included: t-SNE / UMAP views, DBSCAN, and model-based feature attribution (random
  forest, SHAP); effect sizes with FDR control are used instead, which avoids training a
  classifier on labels derived from the same features.
- Position within a vessel uses height in the scan as a proxy, not a centreline.

## License

MIT, see [LICENSE](LICENSE).
