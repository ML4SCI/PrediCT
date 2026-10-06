# D008 Per-artery masks via filter-then-rasterise

**Date**: 2026-06-02
**Stage**: features
**Status**: Active
**Module**: `src/predict/features/per_artery_mask.py`

## Decision

For each canonical vessel (`LAD`, `RCA`, `LCx`, `LM`), build a 3D binary mask by:

1. Filtering the patient's `ParseResult` to ROIs whose `vessel` equals the target.
2. Passing the filtered `ParseResult` plus the same `excluded_roi_ids` set used for the whole-mask to `predict.preprocess.mask_builder.build_3d_mask`.
3. The result is a `(n_slices, H, W)` uint8 mask per vessel.

The function returns a `dict[str, np.ndarray]` keyed by canonical vessel name. Empty vessels return an all-zero mask of the correct shape (downstream skips them).

## Rationale

- **Honest vessel identity**: the patient-level whole-mask discards which voxel belongs to which artery once polygons are filled into a binary array. Recovering vessel labels post-hoc (nearest-ROI labelling) introduces error.
- **Consistency with whole-mask**: using the same exclusion set means per-vessel mask voxels sum (within rounding) to the whole-mask voxel count, so a per-vessel sanity check is trivial.
- **Reuses `mask_builder.build_3d_mask`**: no new rasterisation path. One small filter helper is the only new code.
- **Cheap**: 4 extra rasterisations per patient × 444 patients ≈ 3 minutes total.

## Alternatives considered

- **Propagate vessel labels through the original mask**: would require a per-voxel `int8` label volume instead of a binary mask. Costs ~4× the storage, and `mask_builder` grows a second responsibility. Per-vessel features that need a binary input would then have to extract per-vessel slices from the label volume anyway. Rejected.
- **Nearest-ROI labelling on the whole-mask**: assigns each mask voxel to its nearest ROI's vessel. Cheap but introduces labelling errors near vessel boundaries where ROIs may be drawn close together. Rejected.

## Verified by

- `tests/features/test_per_artery_mask.py`: filter correctness, voxel-count sanity (sum of per-vessel ≈ whole-mask), empty-vessel returns all-zero mask, exclusion-set consistency.
