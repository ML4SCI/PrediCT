# D011 Single Agatston thickness-correction helper

**Date**: 2026-06-02
**Stage**: features
**Status**: Active
**Module**: `src/predict/features/agatston.py` (re-exported, used by `per_vessel_aggregates.py`)

## Decision

The Agatston per-ROI score formula lives in one place:

```python
def agatston_roi_score(area_cm2: float, max_hu: float, slice_thickness_mm: float) -> float:
    if max_hu < AGATSTON_HU_THRESHOLD:           # 130
        return 0.0
    return area_cm2 * 100.0 * density_factor(max_hu) * (slice_thickness_mm / 3.0)
```

`density_factor(max_hu) → {1, 2, 3, 4}` per Agatston 1990. The thickness correction `(slice_thickness_mm / 3.0)` scales a non-3 mm protocol to the 3 mm reference. Both `agatston.py` and `per_vessel_aggregates.py` import this helper; neither re-implements the formula.

## Rationale

Two Agatston implementations with separate thickness handling can diverge silently. On COCA's 3 mm cohort the correction factor is 1.0, so such a bug stays latent; on any non-3 mm cohort (for example 2 mm or 2.5 mm slices) the two would disagree.

The design prevents this class of bug structurally: one helper, one call site per feature path. Adding a new cohort with different slice thickness changes one constant and works everywhere.

## Alternatives considered

- **Two implementations with a regression test**: vulnerable to drift; if the test is forgotten or the implementations are touched independently, divergence reappears. Rejected.
- **Inline the formula at each call site**: same drift risk. Rejected.

## Verified by

- `tests/features/test_agatston.py::test_thickness_correction_unified`: fixture patients at 3.0 mm and 2.5 mm; assert `agatston.compute_agatston` and `per_vessel_aggregates.compute_per_vessel_aggregates` return identical per-vessel Agatston values.
- Single import of `agatston_roi_score` in `per_vessel_aggregates.py` (grep-enforced).

