# D007 Lesion grouping rule

**Date**: 2026-06-02
**Stage**: features
**Status**: Active
**Module**: `src/predict/features/lesion_ccl.py`

## Decision

Group ROIs into 3D lesions per artery using BFS connected components. Two ROIs `i`, `j` are connected if **all** of:

1. Same canonical vessel (`vessel != None` and equal).
2. Both `(image_index, roi_idx_in_slice)` keys are absent from the `excluded_roi_ids` set (D012: round-trip-failed and dirty ROIs are not eligible).
3. Slice-index gap ≤ **`LESION_GROUPING_MAX_SLICE_GAP = 1`** (strict adjacency; absolute difference in CT array slice indices).
4. In-plane Euclidean centroid distance ≤ **`LESION_GROUPING_DISTANCE_MM = 5.0 mm`**.

The slice index comes from `predict.preprocess.slice_matcher.match_roi_to_slice` (Z-coordinate matching, D001). ROIs that are unmatched in Z are skipped (not assigned to any lesion).

Same-slice multiple ROIs in the same artery form separate lesions (the rule requires `|slice_diff| ≤ 1` AND in-plane distance, same-slice means `|slice_diff| = 0` is allowed, so same-slice ROIs within 5 mm merge; same-slice > 5 mm apart stay separate).

## Rationale

The thresholds (5 mm in-plane, gap = 1) are conservative on the **over-splitting** side, which is the safer error direction:

- Hoori 2024 identified **lesion count** as the top-1 MACE predictor. Over-splitting inflates count slightly; under-merging is acceptable. Over-merging directly corrupts the count and the diffusivity denominator.
- 5 mm in-plane captures the geometric upper bound of centroid shift between adjacent 3 mm slices for a single lesion up to ~10 mm transverse diameter (curved artery trajectory). Beyond ~5 mm shift, the polygons would visually look like two distinct deposits to the annotator.
- Gap = 1 (strict adjacency) trusts the radiologist annotation: a gap > 1 means the annotator decided there was no calcium on the intermediate slice.

A planned **sensitivity probe** at thresholds `{3, 5, 8} mm × {gap 1, 2}` on 10 representative patients verifies that 5 mm / gap = 1 produces lesion counts within ±10% of neighbouring settings. If the probe shows large swings, this decision is revisited.

## Alternatives considered

- **3D CCL on rasterised mask** (scipy.ndimage.label): anisotropic voxels (0.5 × 0.5 × 3 mm) mean 6-connectivity fragments single lesions whose pixels shift between slices; 26-connectivity over-merges close deposits. The threshold question resurfaces as a connectivity choice. Rejected.
- **3D centroid Euclidean distance (single threshold)**: combines XY and Z into one number. Loses the "annotator decided no calcium" signal that gap = 1 preserves. Rejected.
- **Polygon overlap**: polygons of the same lesion almost never overlap in 2D projection at 3 mm spacing; this rule would fragment most lesions. Rejected.
- **5 mm centroid + gap = 2**: would bridge a missed annotation slice. Concrete cost: merges two distinct proximal-LAD deposits separated by a calcium-free slice into one "lesion", corrupting count and diffusivity. The annotation-quality benefit is hypothetical; the corruption risk is concrete. Rejected.

## Verified by

- `tests/features/test_lesion_ccl.py`: adjacent close → 1 lesion; adjacent far → 2; gap = 2 → 2; same-slice multiple ROIs → separate; cross-vessel → never merge; dirty/excluded ROIs → not grouped.
- `scripts/probe_lesion_grouping.py`: cohort-level threshold sensitivity (run after first full features extraction).
