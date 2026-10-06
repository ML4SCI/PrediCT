# D039 Slice geometry check on the selected series

**Date**: 2026-10-03
**Stage**: io
**Status**: Active
**Module**: `src/predict/io/dicom_loader.py` (`check_geometry`, `select_series`), `src/predict/io/patient_discovery.py`

## Decision

After the D006 series selection, the selected series must pass a geometry check before any volume, mask or feature is built from it:

1. at least 2 slices, and every file has `ImagePositionPatient` (no stand-in z);
2. every file is axial: `ImageOrientationPatient` within 1e-3 of (1, 0, 0, 0, 1, 0);
3. regular: every z step within 0.05 mm of the median z step (no gaps, no duplicates);
4. one `SliceThickness`, and contiguous: |median z step - thickness| <= 5% of thickness (no overlapping or gapped reconstruction).

A patient that fails is excluded at discovery with reason `irregular_geometry` and the failing check in `outputs/01_manifest/exclusions.csv`. Both loaders raise `IrregularGeometryError` on such a series, so no later stage can build one by accident. Split series are not merged.

Discovery, `load_patient_dicom` and `load_patient_metadata` share one implementation (`select_series` / `read_selected_headers`), so the manifest labels, the CT volume and the header-only metadata describe the same files. The manifest records `series_uid`, `n_slices` and `z_step_mm`. The header-only loader reports the checked z step as its z spacing (what SimpleITK assigns to the volume), not `SliceThickness`.

## Rationale

SimpleITK silently assigns a uniform spacing of (z_last - z_first) / (n - 1) to whatever slice list it is given. A series with missing slices is therefore stretched as a whole, not only near the gap: resampling (`preprocess/resampling.py`) treats the false spacing as true, so every lesion is taller than it is. Lesion grouping (D007) links ROIs by array slice index, so slices 15 mm apart counted as adjacent. The D002 round-trip cannot see this; it compares pixel values per array slice, not positions.

A header-only sweep of all 449 folders (`scripts/diagnose_slice_geometry.py`, `scripts/diagnose_series_detail.py`) found, in the 444 manifest patients:

| pid | finding | SimpleITK z spacing |
|---|---|---|
| 135 | one slice missing (6 mm step); the missing slice is in the folder as a separate 1-slice series on the same grid | 3.068 (true 3.0) |
| 159 | 11 slices missing (steps 15, 15, 6, 9 mm); a second same-grid series fills 10 of them; z -182 is missing everywhere; one file has no position at all; 3 ROIs sit on missing slices (the real cause of its D002 exclusions) | 3.786 (true 3.0) |
| 388 | one slice missing (6 mm step), not in the folder; the folder's 1-slice "scout" is pid 135's slice (same anonymised series UID, same z, 135's grid), i.e. one file is filed under two patients | 3.070 (true 3.0) |
| 33 | 3.0 mm slices every 1.5 mm (overlapping reconstruction; the only B35f scan). Geometry is consistent, but every calcification is sampled about twice, so the Agatston weight (thickness / 3, D011) roughly doubles its score and its per-lesion ROI counts | 1.5 |
| 38 | 2.95 mm steps, 3.0 mm slices (1.7%); geometry consistent | 2.95 |

No series was tilted or had duplicate z. The threshold values are not tuning: COCA z steps are either exactly regular or off by whole missing slices (>= 3 mm), and step / thickness is 1.0, 0.983 or 0.5, so any z tolerance up to ~1 mm and any contiguity tolerance between ~2% and ~50% gives the same cohort. Result: 33, 135, 159, 388 excluded; 440 patients.

## Alternatives considered

- **Merge same-grid split series back into the selected one.** Rejected. It fully repairs only pid 135 (159 and 388 keep a gap), and the file that repairs 135 also sits in pid 388's folder, so the series grouping in these folders is not trustworthy. A second code path to rescue one patient is not worth the risk.
- **Rebuild an irregular series on a regular grid with the missing slices marked or interpolated.** Rejected: interpolation fabricates data inside lesions (159 and 388 have ROIs on the slices bounding a gap); blank slices split lesions and corrupt texture features.
- **Keep pid 33 and weight Agatston by the slice increment instead of the thickness.** Rejected: it fixes the score but not the doubled per-lesion ROI counts or the different partial-volume behaviour of overlapping slices. pid 33 is also the only B35f scan and is already dropped from clustering as a singleton kernel.
- **Keep everything (status quo).** Rejected: four patients enter every stage with wrong geometry or doubled calcium.

## Consequence

Stage 1 now excludes 4 more patients (444 to 440). Expected stage 2 change: pid 159 was one of the 7 D002 partial patients, so 436 / 440 pass fully. Every reported result is computed after this exclusion.

## Verified by

- `tests/io/test_dicom_loader.py`: `test_geometry_*` (missing slice, duplicate z, overlap, 2.95 / 3.0 accepted, missing position, tilt, single slice, mixed thickness).
- `tests/io/test_patient_discovery.py`: `test_discovery_excludes_irregular_geometry`, `test_loaders_refuse_irregular_geometry`, `test_loaders_agree_on_z_spacing`, `test_discovery_uses_configured_xml_dir`.
- `tests/io/test_config.py::test_data_raw_must_be_named_raw`.
- Rerun of stage 1: `exclusions.csv` lists exactly 33, 135, 159, 388 as `irregular_geometry`.
