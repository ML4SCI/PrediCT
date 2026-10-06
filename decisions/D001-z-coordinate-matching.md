# D001 Z-coordinate matching as primary ROI→slice mapping

**Date**: 2026-06-01
**Stage**: io / preprocess
**Status**: Active
**Module**: `src/predict/preprocess/slice_matcher.py`

## Decision

Each XML ROI is matched to a DICOM slice by Z-coordinate: take the ROI's
`Center[2]` (world Z in mm) and find the slice whose `ImagePositionPatient[2]`
is closest. Match is rejected if `|delta| > 1.5 mm` (half the native 3 mm slice
thickness).

`ImageIndex` math is fallback only and uses **direct mapping**
(`slice_idx = ImageIndex`), with no inversion. The fallback is used only when
the ROI lacks a `Center` field.

## Rationale

Empirically verified on patient 306 (high-burden, Agatston-positive):
ImageIndex=15 ROI with XML `Mean=156.2`, `Max=182.0`.

| Mapping | Slice voxel Mean | Slice voxel Max |
|---|---|---|
| Direct (`slice_idx = ImageIndex`) | 122.8 | **182.0** (matches XML) |
| Flipped (`slice_idx = n - 1 - ImageIndex`) | 11.4 | 56.0 (soft tissue) |

The XML stat round-trip succeeds with direct mapping and fails with flipped.

Z-coordinate matching is preferred over `ImageIndex` math because it is robust
to the multi-series and duplicate-Z edge cases described in the dataset report
(sections 7.4 A–C). It uses physical position from DICOM metadata, which is
authoritative per the DICOM standard, rather than the OsiriX/Horos internal
counter exposed in the XML.

## Alternatives considered

- **Flipped index** (`slice_idx = (n_slices - 1) - ImageIndex`): empirically wrong;
  produces Z-flipped masks. Rejected.
- **`InstanceNumber - 1` mapping**: matches ~50% of patients per dataset
  report § 7.3; scanner-dependent. Rejected.
- **Filename sort**: ~50% match rate. Rejected.

## Verified by

- `tests/preprocess/test_slice_matcher.py`
- `tests/preprocess/test_mask_builder.py::test_xml_roundtrip_on_fixture`
