# D010 `low_burden_flag` for very-small-mask patients

**Date**: 2026-06-02
**Stage**: features
**Status**: Active
**Module**: `scripts/03_features.py`

## Decision

Add a boolean column `low_burden_flag` to `outputs/03_features/features.csv`:

```
low_burden_flag = (mask_voxels < 100)
```

`mask_voxels` is the patient's whole-mask voxel count from stage 2 (`outputs/02_preprocessed/preprocess_report.csv`). The threshold (100 voxels) is from config: `features.low_burden_voxel_threshold`.

**No feature is dropped or set to NaN at extraction time** based on this flag. Every patient gets a full feature row. The flag is metadata for downstream filtering.

## Rationale

PyRadiomics' `minimumROISize=14` already prevents extraction on the tiniest masks. But masks in the 14–100 voxel range pass that gate and produce numerically defined texture features whose biological meaning is questionable, too few unique gray levels after binning, degenerate co-occurrence matrices.

Two design options were considered:

1. **Gate at extraction time** (NaN texture for low-burden patients): produces patient-specific NaN patterns that complicate downstream PCA / clustering / ICC. Loses data when it's not strictly needed.
2. **Track but don't filter** (D010): every patient gets a complete row; downstream code that wants to exclude low-burden patients filters on one column. Reduce/analyse stages run a planned "sensitivity-exclusion" rerun (re-run phenotype discovery excluding `low_burden_flag == True`, confirm clusters are stable) as a published robustness check.

Option 2 is more robust:
- Avoids NaN-induced PCA degeneracy.
- Avoids losing data unnecessarily: many features (Agatston, lesion count, per-artery aggregates) are voxel-count-independent and remain valid at any burden.
- The ICC perturbation gate in stage 4 separately handles cohort-wide feature reliability: texture features that are systematically unstable on small masks fail ICC and get dropped from the analysis set anyway.

## Alternatives considered

- **Gate at extraction time (set texture columns to NaN)**: rejected for the reasons above.
- **Drop low-burden patients entirely from analysis**: loses data; low-burden patients are still part of the cohort and Agatston / spatial features for them are valid.
- **Higher threshold (e.g., 200, 500)**: too aggressive; would flag many legitimate low-burden patients whose first-order and shape features are fine. 100 is the IBSI-suggested "should be reliable" floor for full texture; below it, texture is suspect.

## Graceful degrade for masks below PyRadiomics' minimumROISize

A subset of `low_burden_flag == True` patients have masks below the IBSI floor
(`minimumROISize = 14` per D009). PyRadiomics refuses to extract on these and
raises `ValueError: Size of the ROI is too small`.

The orchestration script catches this exception, sets:

  - `radiomics_status = "skipped"`,
  - `radiomics_reason = "<PyRadiomics error message>"`,

and writes the row anyway with the XML-derived features (Agatston, spatial,
density tiers, per-vessel aggregates) intact. The PyRadiomics columns are
absent for that row and become `NaN` in `features.csv` once pandas pads the
schema.

Empirical result on the 444-patient cohort: 22 patients skip PyRadiomics
(masks 6–14 voxels), 422 get a complete row. All 22 are also flagged with
`low_burden_flag = True`.

Rationale for this design: dropping those 22 patients would lose perfectly
valid XML-derived features for the lowest-burden segment of the cohort.
Lowering `minimumROISize` would compute texture on noise. Keeping the rows
with NaN'd PyRadiomics is the only option that preserves data without
silently introducing meaningless texture values.

## Verified by

- `scripts/03_features.py` writes the column for every patient.
- Full-cohort run: 444 rows in `features.csv`, 22 with NaN PyRadiomics
  columns and `radiomics_status="skipped"`.
- Reduce stage carries the flag through and runs the sensitivity-exclusion
  analysis (logged in stage 5 doc).
