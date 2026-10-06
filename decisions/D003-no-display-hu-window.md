# D003 No display HU window in pipeline outputs

**Date**: 2026-06-01
**Stage**: preprocess
**Status**: Active
**Module**: `src/predict/preprocess/hu_handling.py`

## Decision

Preprocessing emits **one** CT volume per patient: raw HU values, clipped to
`[clip_min, clip_max]` from config (default `[-200, 3000]`). No
`*_ct_display.npy` (windowed/normalised) file is saved.

## Rationale

PyRadiomics and downstream feature extraction require raw HU. The display
window (for example HU center 300, width 1500) is a visualisation convenience
that no downstream module reads. Saving it would double the storage cost of
preprocessing outputs (>20 GB on the full cohort).

## Alternatives considered

- **Keep display window for QA**: any visualisation can be windowed on the
  fly from the raw volume. No storage gain to pre-computing it. Rejected.

## Verified by

- `tests/preprocess/test_hu_handling.py`
- File listing in `outputs/02_preprocessed/` has exactly two files per
  patient: `{pid}_ct.npy`, `{pid}_mask.npy`.
