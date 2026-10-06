# D002 XML stat round-trip as preprocessing correctness gate

**Date**: 2026-06-01
**Stage**: preprocess / validate
**Status**: Active
**Module**: `src/predict/validate/xml_roundtrip.py`

## Decision

For every non-dirty ROI (`vessel is not None`, `n_points > 0`), after the
3D mask is built, the round-trip checker re-rasterises the polygon on the
matched CT slice and compares voxel statistics against the XML-stored
`Mean`, `Max`, `Min`.

Per-ROI pass criteria:

- **Hard gate**: `|voxel_max - xml_max| == 0` (exact match). This is the
  slice-correctness signal: the brightest voxel inside the polygon on the
  matched slice is the same one OsiriX saw.
- **Informational**: `|voxel_mean - xml_mean|` is recorded but does not
  cause failure unless it exceeds 200 HU (gross mismatch). See *Rationale*.

Per-patient pass criteria: 100% of non-dirty ROIs must clear the hard gate.
A failing patient is logged with the offending ROI list and excluded from
downstream stages.

## Rationale

The XML stores per-ROI HU statistics computed by OsiriX/Horos at the time
of annotation. Empirically (smoke test, 5 patients, 232 ROIs):

- `delta_max = 0` on every ROI when slice mapping is correct.
- `delta_mean` is systematically negative (voxel mean 20–60 HU below XML
  mean) on every ROI, every patient.

The systematic mean offset is a **binary-vs-partial-volume rasterisation
artefact**, not a pipeline bug. OsiriX renders the float polygon with
anti-aliased / partial-volume weighting: boundary pixels contribute to the
mean weighted by their fractional coverage. We use a binary mask (a pixel
is in or out), as does PyRadiomics and every standard radiomics tool. The
binary mask includes whole boundary pixels at lower HU (calcium-to-soft-
tissue gradient), pulling the mean down. The effect is consistent across
all patients.

`cv2.fillPoly`'s `shift` parameter does not close the gap: it adjusts
polygon edge geometry but the pixel-inclusion rule remains center-based
and integer-quantised, so the result is identical to int-rounded vertices
for our polygons.

Max HU is unaffected: the brightest single pixel inside the polygon is the
same whether weighting is binary or partial-volume.

Therefore the round-trip's hard gate is **Max HU exactness**, which is the
direct correctness signal for slice mapping and polygon shape. Mean is
informational; persistent large deltas would signal a different problem
(wrong polygon coordinates entirely) but do not invalidate the binary mask
used by downstream feature extraction.

## Alternatives considered

- **Exact Mean match**: not achievable without replicating OsiriX's
  partial-volume rasterisation, which would not change our downstream
  feature extraction (PyRadiomics consumes a binary mask). Rejected as
  unnecessary engineering.
- **Implement partial-volume weighting in the round-trip checker only**:
  doable (super-sample each boundary pixel at e.g. 8×8 and count fractional
  coverage) but does not improve pipeline correctness; the binary mask we
  build and feed forward stays the same. Deferred unless a future
  application requires OsiriX-equivalent stats.
- **No round-trip check**: a Z-flipped slice mapping (D001) would go
  undetected through every later stage. Rejected.

## Verified by

- `tests/validate/test_xml_roundtrip.py`
- Invoked at the end of `scripts/02_preprocess.py` for every patient; aggregate
  report in `outputs/02_preprocessed/xml_roundtrip.csv`.
