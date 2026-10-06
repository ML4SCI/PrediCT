# PrediCT: Coronary Artery Calcium Radiomics Phenotyping

Endpoint-free, reproducible analysis of coronary artery calcium (CAC) structure in non-contrast
gated cardiac CT, built for **Google Summer of Code 2026 (ML4SCI)** on the public **Stanford COCA**
cohort (449 scans with expert polygon calcium annotations; 436 patients in the analysis cohort
after the exclusions listed in [docs/data.md](docs/data.md)).

The question: *do calcified coronary arteries fall into discrete morphological phenotypes once
total calcium burden is accounted for?* It is answered without any clinical outcome label, by
combining radiomics feature extraction, perturbation-based reliability gating, unsupervised
clustering, and stability-based validation, with every analysis rule written down in
[decisions/](decisions/) before the corresponding result was computed.

## Main results

- **No stable patient-level subtypes beyond burden.** Total calcium burden dominates every
  patient feature space examined. The one reproducible patient split (spatial features, k=2,
  Hennig bootstrap Jaccard 0.84 to 0.94) is a burden dichotomy (Cliff's delta -0.90 for Agatston
  between the two groups; D024), and an external scanner holdout is assigned by burden.
- **Standard cluster-number tools are not diagnostic here.** On a strongly skewed burden axis the
  gap statistic and taxometric analysis fail their planted-structure and continuum controls
  (D036.1, D038), so the analysis claims neither a continuum nor discrete types; gap results are
  reported descriptively (D021.1).
- **A robust lesion-level signature of advanced disease.** Lesion clustering finds a core of
  large, dense, multi-slice, RCA-dominant plaques with no left-main lesions, recovered at every k
  from 7 to 12, within each scanner-kernel stratum, and under annotation perturbation
  (membership Jaccard 0.94). Its carriers have far higher total burden; whether it carries
  information beyond total burden could not be established
  ([experiments/lesion_morphology/findings.md](experiments/lesion_morphology/findings.md)).
- **Scanner kernel association is explained by burden** (D021.1): the kernel-by-cluster
  association does not survive adjustment for Agatston score.

All statements are conditional on this single cohort, which has no outcome data.

## Pipeline

```
io -> preprocess -> features -> stability -> reduce -> discover -> analyse -> validate
   -> exploratory lesion analyses -> figures
```

| Stage | Purpose |
|---|---|
| 1 `io` | Patient discovery, DICOM series selection, slice-geometry checks, calcium-XML parsing |
| 2 `preprocess` | Z-coordinate ROI-to-slice matching, 3D mask rasterisation, resampling to 0.375 x 0.375 x 3.0 mm, XML-statistic round-trip check |
| 3 `features` | Agatston scoring, 3D lesion grouping, per-vessel and spatial features, density tiers, PyRadiomics shape and texture |
| 4 `stability` | Realistic acquisition and annotation perturbations (primary, D033) and a deterministic set (supplementary, D014); ICC reliability gate |
| 5 `reduce` | Variance filtering, ComBat kernel harmonisation, Spearman redundancy clustering, PCA |
| 6 `discover` | Hopkins statistic, gap statistic, consensus clustering, Hennig bootstrap stability, kernel association |
| 7 `analyse` | Per-cluster characterisation, burden orthogonality, pre-registered directional hypotheses, cross-cohort consistency |
| 8 `validate` | External scanner (GE) holdout, leave-k-out cross-validation, cross-cohort ARI |
| 9 exploratory | Pre-registered lesion-level analyses in `experiments/` (lesion morphology, LAD phenotype, isolated left-main disease) |
| 10 figures | Paper figures from the published stage outputs |

Every stage writes to a temporary folder and publishes it in one step together with a completion
record (input hashes, code fingerprint, output hashes). Each consumer verifies that record before
reading, so a partial or stale result is never used.

## Repository layout

```
configs/      YAML configuration (paths, seeds, thresholds, perturbations)
params.yaml   PyRadiomics extraction settings
src/predict/  Library code, one subpackage per stage
scripts/      Stage scripts (numbered) and diagnostic / probe scripts cited by the decisions
tests/        pytest suite mirroring src/predict/
experiments/  Pre-registered exploratory analyses, each with a plan and findings
decisions/    Design-decision registry, one markdown per decision
docs/         Per-stage documentation, dataset report, reproducibility notes
```

`data/` and `outputs/` are not tracked.

## Data

This repository contains **code only**. The Stanford COCA (Coronary Calcium) dataset is available
from the Stanford AIMI shared datasets and is not redistributed here. Place the DICOM series and
calcium XML annotations at the paths configured in `configs/default.yaml` (`data/raw/`,
`data/calcium_xml/`) to run the pipeline end to end.

## Reproducibility

- Every stochastic step takes an explicit seed; parallel bootstraps derive per-iteration seeds,
  so results do not depend on the number of workers.
- Stages communicate through files bound by completion records and can be rerun independently
  (`run_pipeline.sh --from N --to M`).
- Unit and integration tests: `pytest tests/` and
  `pytest experiments --import-mode=importlib`.
- Every design choice, threshold, and interpretation rule is an individual entry in `decisions/`.

## Getting started

```bash
# Python 3.8, conda recommended
conda install -c radiomics -c conda-forge pyradiomics
pip install -r requirements.txt
pip install -e .

pytest tests/                    # test suite
bash run_pipeline.sh             # stages 1-10 (requires the dataset in place)
bash run_pipeline.sh --from 6    # rerun from stage 6
```

Worker count comes from `PREDICT_N_JOBS`, or the scheduler's CPU allocation (`PBS_NCPUS`,
`NCPUS`, `SLURM_CPUS_ON_NODE`). `submit_pipeline.sh` is a PBS Pro wrapper.

## License

MIT, see [LICENSE](LICENSE).
