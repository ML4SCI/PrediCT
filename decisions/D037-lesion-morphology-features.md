# D037 Lesion morphology clusters on four non-degenerate features

**Date**: 2026-10-02
**Stage**: exploratory (lesion morphology)
**Status**: Active (fixes a construction defect in the lesion feature matrix)
**Module**: `experiments/lesion_morphology/run.py` (`build_morphology_matrix`)

## Decision

The lesion-level k-means clustering uses four z-scored features: `log1p_volume`, `mean_hu_weighted`, `max_hu`, `log1p_n_rois`. Two former features are removed. A morphology column whose SD is below 1e-9 x max(1, |mean|) now raises instead of being scaled.

## Rationale

`src/predict/features/lesion_ccl.py:216` defines `volume_mm3 = total_area * slice_thickness_mm`. Therefore:

- `log1p_volume_area_ratio` is log1p(slice thickness). In COCA it is 3.0 mm for every lesion (one unique value). Z-scoring it divided the rounding noise of this constant by a near-zero SD, producing a column with 4 distinct values, SD 0.466, range -1.81 to 0.90: noise entered the clustering as a sixth feature.
- `log1p_area` and `log1p_volume` are deterministic functions of each other (volume = 3 x area), so lesion size was weighted twice.

Found by `experiments/lesion_morphology/lesion_stability_d033.py`: the ratio feature had ICC 0.36 under D033 M1 jitter (40-patient smoke), while every real feature had ICC >= 0.997. The removal follows from the code that builds the features, not from any clustering outcome.

## Consequence

Every lesion-level result is computed on the four-feature matrix: run.py (the gap-selected k is used as the primary k), analyse, finalise, figures, the k sweep, the C8 other-lesions and burden-matched tests, the LAD and LM-isolated experiments, and the lesion stability check. All pre-registered rules carry over unchanged (k sweep 7..12; other-lesions; burden-matched; lesion stability).

## Verified by

- `lesion_stability_d033.py` on the four features (ICC per feature, C8 Jaccard).
- The near-constant guard in `build_morphology_matrix`.
