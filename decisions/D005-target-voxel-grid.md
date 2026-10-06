# D005 Target voxel grid for resampling

**Date**: 2026-06-01 (in-plane target set to 0.375 mm on 2026-09-01)
**Stage**: preprocess
**Status**: Active
**Module**: `src/predict/preprocess/resampling.py`
**Config**: `configs/default.yaml` → `resample.in_plane_mm`, `resample.slice_mm`

## Decision

Resample CT and mask to a fixed voxel grid: **`0.375 × 0.375 × 3.0 mm`**.

- **In-plane (0.375 mm)**: the cohort **native median** is 0.377 mm
  (range 0.26 to 0.71). Targeting 0.375 mm is therefore the
  minimum-interpolation choice: roughly half the cohort moves by less than
  a hundredth of a millimetre, and no patient is resampled further than
  necessary to reach a common grid.
- **Z (3.0 mm)**: preserves native slice thickness for 99% of the cohort.
  The 4 GE patients at 2.5 mm are excluded by D004; if reintroduced they
  would upsample slightly in z.

## Rationale

A common grid is required so that PyRadiomics shape and texture features
are comparable across patients. The grid should interpolate as little as
possible: a target above the native median (for example 0.5 mm) would
systematically downsample most of the cohort, and a target below it would
upsample most of it. The native median minimises the total resampling, and
this argument does not depend on any downstream result.

## Anisotropy note

This grid is **not isotropic**: the z spacing is **8× the in-plane**.
Texture features (GLCM/GLSZM/GLRLM/NGTDM/GLDM) computed in 3D therefore
weight in-plane voxel pairs differently from z-pairs. This is a limitation
of NCCT calcium radiomics at 3 mm slice thickness and is documented in the
features-stage notes. The function is named `resample_to_target` (not
`to_isotropic`) for this reason.

## Alternatives considered

- **True isotropic (1.0 × 1.0 × 1.0)**: requires synthesising z data
  between 3 mm slabs. Texture features would reflect interpolation
  artefacts more than real biology. **Rejected**; `slice_mm` stays at 3.0.
- **Native (no resample)**: PyRadiomics shape and texture features become
  patient-specific by spacing, not comparable across the cohort.
  **Rejected**.
- **0.5 × 0.5 × 3.0**: a round-number common grid, but above the native
  median, so most of the cohort would be downsampled. **Rejected** in
  favour of the native median.
- **0.4 × 0.4 × 3.0**: also close to the native median and defensible;
  0.375 is closer.

## Verified by

- `tests/preprocess/test_resampling.py`
