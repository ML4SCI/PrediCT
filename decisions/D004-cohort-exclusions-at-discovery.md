# D004 Cohort exclusions applied at patient discovery

**Date**: 2026-06-01
**Stage**: io
**Status**: Active
**Module**: `src/predict/io/patient_discovery.py`

## Decision

`discover_patients()` reads the exclusion list from `configs/default.yaml`
(`cohort.exclude_pids` and `cohort.exclude_ge_scanners`) and returns the
manifest with exclusions already applied. Downstream stages receive only the
eligible cohort.

Default exclusions (from the dataset analysis report):

- **PID 12, 197**: XML present, no DICOM data.
- **PID 268**: DICOM series replaced (1.5 mm → 3.0 mm); XML `ImageIndex`
  values exceed current slice count; spatial accuracy of annotations lost.
- **GE scanners (4 patients)**: `Manufacturer != "SIEMENS"`; STANDARD kernel
  + 2.5 mm slices differ materially from the Siemens 99% cohort. Held out
  as an external check rather than included in primary analysis.

## Rationale

Centralising exclusions at discovery prevents silent inclusion of bad data
downstream. Without them the pipeline would run on PID 12 / 197 (no DICOM,
exception-driven failure) and PID 268 (1.5 mm annotation series silently
producing wrong masks).

## Alternatives considered

- **Per-stage exclusion checks**: error-prone, easy to miss in a new stage.
  Rejected.
- **Soft exclusions (warn but include)**: failures propagate as
  feature-matrix `NaN`s and corrupt aggregate statistics. Rejected.

## Verified by

- `tests/io/test_patient_discovery.py`
- `outputs/01_manifest/manifest.csv` row count matches `442` after default
  exclusions on the 449-patient COCA cohort.
