# D012 `excluded_roi_ids` as explicit input to every feature path

**Date**: 2026-06-02
**Stage**: features
**Status**: Active
**Module**: every module under `src/predict/features/`

## Decision

Every feature-extraction function that consumes a `ParseResult` takes an `excluded_roi_ids: set | None = None` keyword argument and, when non-empty, skips ROIs whose `(image_index, roi_idx_in_slice)` tuple is in the set.

The orchestration script `scripts/03_features.py` loads `outputs/02_preprocessed/xml_roundtrip.csv` once at start, derives the per-patient exclusion set, and passes it to:

- `compute_agatston(...)`
- `group_rois_into_lesions(...)`
- `compute_per_vessel_aggregates(...)`
- `compute_density_tiers(...)`
- `build_per_artery_masks(...)`

Dirty-vessel ROIs (`roi.vessel is None`) are *separately* skipped inside each module, D012 covers the round-trip-failure exclusion, the parser-time dirty filter is a different layer.

## Rationale

In stage 2 we decided that ROIs failing the Max-exact round-trip gate are excluded from the saved mask. For features to be consistent with the mask, every feature path must apply the same exclusion. Without this, a patient's XML-derived Agatston (which would still see the failing ROIs) would disagree with their PyRadiomics features (which see the cleaned mask), and downstream the row is internally incoherent.

Making the exclusion set an explicit input rather than a module-level constant or hidden state keeps the contract crystal: every feature value in a row is computed from the same input set.

## Alternatives considered

- **Filter the `ParseResult` once at the top and pass the filtered version everywhere**: would also work, and is in some ways simpler. Rejected because:
  - It rebuilds the `ParseResult` dataclass with a filtered `slices` field, which changes `roi_idx_in_slice` numbering (the index within each slice's `rois` tuple). The exclusion set was computed against the *original* indices; renumbering them is error-prone.
  - Module signatures still need to be tested with both empty and non-empty exclusion sets: the explicit-input form makes that obvious.
- **Module-level exclusion state**: anti-pattern. Forces module-level mutation per patient inside a multi-process worker. Rejected.

## Verified by

- `tests/features/test_*` for each module: each has a test that an excluded ROI does not contribute (Agatston unchanged, lesion grouping skips it, density tier count drops, per-vessel volume drops, per-artery mask doesn't rasterise it).
- Orchestration test: a patient with 1 excluded ROI has the SAME features whether passed `excluded_roi_ids={(img_idx, roi_idx)}` or with that ROI manually removed from the source XML.
