# Cohort and Data

Source: Stanford COCA, gated non-contrast cardiac CT with expert XML calcium annotations.

## Counts and exclusions

| Step | Patients | Removed | Rule |
|---|---|---|---|
| DICOM folders in `raw/` | 449 | | XML files: 451; pids 12 and 197 have XML but no DICOM and are never part of the 449 |
| Stage 1 discovery | 439 | 268 (replaced series, 1.5 mm vs 3.0 mm mismatch; `configs/default.yaml`) | D004 |
| | | 19, 28, 76, 77 (GE scanners, held out for external validation) | D004, D029 |
| | | 33, 135, 159, 388 (irregular or overlapping slice geometry) | D039 |
| | | 156 (mixed in-plane grid across slices) | D039.1 |
| Stage 3 PyRadiomics | 437 | 2 patients whose whole mask is at or below PyRadiomics' `minimumROISize` (14 voxels); their XML-derived features are kept | D009, D015 |
| Stage 5 analysis matrix | **436** | 1 patient with a singleton scanner kernel (I36f/3), which ComBat cannot harmonise | D019 |

Kernels in the analysis cohort: Qr36d/2 231, I30f/3 205. Kernel is a confound and is
harmonised by ComBat in `predict.reduce.prepare_matrix` (D019, D035); kernel-stratified
subcohorts are analysed separately as replication cohorts.

Lesion-level analyses (stage 9) cluster the lesions of all 439 stage 1 patients and report
patient-level comparisons on the 436-patient cohort.

## Acquisition
- Slice thickness: 3.0 mm
- KVP: 120 for almost all scans
- Image dim: 512 x 512
- Native pixel spacing: 0.26 to 0.71 mm (median about 0.38); resampled to 0.375 x 0.375 x 3.0 mm (D005)

## Annotations
- Format: Apple plist XML (one per patient)
- Vessels: LAD, RCA, LCx, LM (left main)
- Non-standard vessel names in patients 238, 398, 415 and 421 are normalised in `predict.io.xml_parser`.

## Class distribution
- Zero-calcium patients: **0**. Every patient has at least one annotated calcium deposit, so the
  analytical question is sub-structure within calcified disease, not burden detection.

## Demographics and outcomes
- Sex, age, race: not in the dataset; no demographic confound checks are possible.
- No clinical outcomes; no result here is validated against events.
