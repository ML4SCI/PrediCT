# D006 Multi-series DICOM folder reduction rule

**Date**: 2026-06-02
**Stage**: io
**Status**: Active
**Module**: `src/predict/io/dicom_loader.py`: `_select_single_series`

## Decision

When a patient's DICOM folder contains more than one ``SeriesInstanceUID``:

1. Filter to ``Modality == "CT"``.
2. Pick the series with the **largest slice count**.
3. Tiebreak: lowest ``SeriesNumber``.
4. Final tiebreak: lexicographic ``SeriesInstanceUID``.

A WARNING is emitted listing all candidate series with slice counts and the
selection marker.

## Rationale

"Lowest SeriesNumber" as the primary key selects the wrong series in at least two COCA cases:

- **Patient 388**: SeriesNumber 2 is a 1-slice scout/preview; SeriesNumber 3
  is the 44-slice diagnostic gated CAC scan. Lowest SeriesNumber selects
  the scout: no annotations match, all 22 ROIs are unmatched.
- **Patient 159**: one DICOM file has missing ``SeriesInstanceUID`` and
  malformed metadata; under the loader's empty-UID fallback it groups alone as a 1-file "series". Lowest SeriesNumber selects
  this broken group; SimpleITK then errors out trying to read pixel data.

The annotated diagnostic scan is reliably the longest series in the folder
across COCA. Sorting by slice count gives the right answer for both 388
and 159, and keeps the SeriesNumber tiebreak for patient 78 (two equal-length
series).

## Alternatives considered

- **Lowest SeriesNumber**: empirically wrong on 388 and 159. Rejected.
- **Cross-reference XML SOPInstanceUID**: COCA plist XMLs do not store
  SOP references; not possible.
- **Filter to series ≥ N slices** (e.g. ≥10): brittle. A patient with a
  short legitimate gated scan plus a longer non-CT or non-annotated study
  would still need the slice-count or tiebreak rule. Subsumed.
- **Latest AcquisitionTime**: not conservative; a re-scan saved later may
  not be the annotated one.

## Verified by

- `tests/io/test_dicom_loader.py::test_multi_series_picks_largest_slice_count`
- `tests/io/test_dicom_loader.py::test_multi_series_equal_slice_count_picks_lower_series_number`
- `tests/io/test_dicom_loader.py::test_multi_series_final_tiebreak_by_uid_lex`
- Empirical: patients 78, 159, 388 load the correct series under this rule.

