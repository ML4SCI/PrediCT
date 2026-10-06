"""DICOM series loading for the COCA cohort.

Loads a patient's CT series into a SimpleITK image, sorted by physical Z
coordinate (ascending). Multi-series folders are reduced to a single series
by the rule in D006, and the chosen series must pass the D039 geometry check
(regular, contiguous, axial) before anything is built from it.

:func:`select_series` is the one place that picks and checks a series.
Discovery and both loaders call it, so the manifest, the CT volume and the
header-only metadata always describe the same files.

The loader returns a :class:`LoadedPatient` carrying everything downstream
stages need:

- the CT image
- per-slice Z positions (so ROIs can be matched by Z; see D001)
- scanner / kernel metadata for kernel harmonisation
- pixel spacing and slice thickness

No HU manipulation, no resampling, no mask handling: those belong to
``predict.preprocess``.

Decisions referencing this module:
    D001: Z-coordinate matching (provides slice_positions)
    D004: Cohort exclusions (kernel/manufacturer used by discovery)
    D039: Slice geometry check on the selected series
"""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

_log = logging.getLogger(__name__)

# D039 thresholds, fixed before the clean rerun. COCA z steps are either
# exactly regular or off by whole missing slices (>= 3 mm), and spacing over
# thickness is 1.0, 0.983 or 0.5, so any value between ~0.02 and ~0.5 gives
# the same cohort (scripts/diagnose_slice_geometry.py).
Z_STEP_TOL_MM = 0.05          # max |z step - median z step|
CONTIGUITY_REL_TOL = 0.05     # max |median z step - thickness| / thickness
AXIAL_TOL = 1e-3              # max |ImageOrientationPatient - (1,0,0,0,1,0)|
# D039.1 (in-plane grid, added after pid 156 passed D039 at discovery and was
# rejected by the loader's SimpleITK spacing check): one in-plane grid per
# series. A volume has one PixelSpacing, one matrix and one (x, y) origin.
PIXEL_SPACING_TOL_MM = 1e-4   # max |PixelSpacing - first slice's|
IN_PLANE_ORIGIN_TOL_MM = 0.05  # max |IPP (x, y) - first slice's|
_AXIAL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class IrregularGeometryError(ValueError):
    """The selected series fails D039 and must not be built into a volume."""


@dataclass(frozen=True)
class LoadedPatient:
    """Result of loading one patient's CT series."""

    pid: str
    ct_sitk: sitk.Image
    slice_positions: tuple[float, ...]  # IPP[2] per array slice, ascending
    pixel_spacing: tuple[float, float, float]  # (x, y, z) mm
    slice_thickness: float
    n_slices: int
    scanner_model: str
    kernel: str
    manufacturer: str
    series_uid: str


def find_dicom_subfolder(patient_dir: Path) -> Path:
    """Return the folder under ``patient_dir`` that contains the .dcm files."""
    for entry in patient_dir.iterdir():
        if entry.is_dir() and any(f.name.endswith(".dcm") for f in entry.iterdir()):
            return entry
    if any(f.name.endswith(".dcm") for f in patient_dir.iterdir()):
        return patient_dir
    raise FileNotFoundError(f"No DICOM files found under {patient_dir}")


def _select_single_series(
    file_meta: list[dict],
    pid: str,
) -> list[dict]:
    """Reduce a multi-series folder to one series.

    Rule (D006):
      1. Filter to ``Modality == "CT"``.
      2. Pick the series with the **largest slice count** (the annotated
         diagnostic acquisition; scouts/previews are typically 1 slice).
      3. Tiebreak: lowest ``SeriesNumber``.
      4. Final tiebreak: lexicographic ``SeriesInstanceUID``.

    Emits a warning via the module logger if the input was multi-series.
    """
    ct_meta = [m for m in file_meta if str(m["modality"]).upper() == "CT"]
    if not ct_meta:
        _log.warning(
            "Patient %s: no DICOM files with Modality=CT; "
            "falling back to all %d files.",
            pid, len(file_meta),
        )
        ct_meta = file_meta

    groups: dict[str, list[dict]] = defaultdict(list)
    for m in ct_meta:
        groups[m["series_uid"]].append(m)

    if len(groups) == 1:
        return ct_meta

    # Sort: most slices first (desc), then lowest SeriesNumber (asc), then UID (asc).
    summary = sorted(
        ((-len(g), g[0]["series_number"], uid) for uid, g in groups.items()),
    )
    chosen_uid = summary[0][2]

    _log.warning(
        "Patient %s: multi-series folder (%d series). "
        "Selecting series with most slices.",
        pid, len(groups),
    )
    for neg_n, snum, uid in summary:
        marker = "  <-- SELECTED" if uid == chosen_uid else ""
        _log.warning(
            "  Series #%s  uid=%s  (%d slices)%s", snum, uid, -neg_n, marker,
        )

    return groups[chosen_uid]


def _normalise_kernel(value) -> str:
    """Flatten pydicom multi-valued ConvolutionKernel into a stable string.

    Siemens encodes the kernel as a backslash-separated multi-value (e.g.
    ``Qr36d\\2``), which pydicom exposes as a list-like ``MultiValue``. We
    join with ``/`` for a deterministic single-token kernel id used in the
    manifest and kernel-harmonisation grouping.
    """
    if value is None:
        return ""
    try:
        # MultiValue is iterable; a plain string is also iterable but is
        # excluded by the isinstance check below.
        if isinstance(value, str):
            return value.strip()
        parts = [str(v).strip() for v in value]
        return "/".join(p for p in parts if p)
    except TypeError:
        return str(value).strip()


def _read_header(path: str) -> dict:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    return {
        "path": path,
        "instance_number": int(getattr(ds, "InstanceNumber", 0)),
        # NaN, never a stand-in, when the position is missing: D039 rejects a
        # selected series containing such a file (pid 159 has one, in a
        # series D006 does not select).
        "z_position": float(ipp[2]) if ipp is not None else math.nan,
        "ipp_xy": (float(ipp[0]), float(ipp[1])) if ipp is not None else None,
        "iop": tuple(float(v) for v in iop) if iop is not None else None,
        "pixel_spacing": (tuple(float(v) for v in ds.PixelSpacing)
                          if hasattr(ds, "PixelSpacing") else None),
        "matrix": (int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))),
        # Some COCA DICOMs are missing SeriesInstanceUID (patient 159). Use
        # an empty-string fallback so all files in the folder that share the
        # missing tag are grouped as a single series by _select_single_series.
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "series_number": int(getattr(ds, "SeriesNumber", 0)),
        "modality": str(getattr(ds, "Modality", "CT")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")).strip(),
        "scanner_model": str(getattr(ds, "ManufacturerModelName", "")).strip(),
        "kernel": _normalise_kernel(getattr(ds, "ConvolutionKernel", "")),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0.0)) or None,
    }


def check_geometry(headers: list[dict], pid: str) -> float:
    """D039: return the z step of a selected series, or raise.

    The series must have >= 2 slices, a position and an axial orientation on
    every file, one in-plane grid (same PixelSpacing, matrix and (x, y)
    origin on every file; D039.1), z steps all within ``Z_STEP_TOL_MM`` of
    their median (no gaps, no duplicates), one slice thickness, and a median
    step within ``CONTIGUITY_REL_TOL`` of that thickness (no overlapping or
    gapped reconstruction). SimpleITK silently assigns a uniform spacing of
    (z_last - z_first) / (n - 1) to whatever it is given, and measures it in
    3D, so an in-plane origin shift also corrupts the z spacing; this check
    must run before the volume is built.
    """
    if len(headers) < 2:
        raise IrregularGeometryError(f"pid {pid}: {len(headers)} slice(s) in the selected series")
    n_no_pos = sum(1 for h in headers if math.isnan(h["z_position"]))
    if n_no_pos:
        raise IrregularGeometryError(
            f"pid {pid}: {n_no_pos} file(s) without ImagePositionPatient")
    for h in headers:
        iop = h.get("iop")
        if iop is None or len(iop) != 6 or max(
                abs(a - b) for a, b in zip(iop, _AXIAL)) > AXIAL_TOL:
            raise IrregularGeometryError(
                f"pid {pid}: non-axial or missing orientation {iop} in {h['path']}")

    ref = headers[0]
    if ref.get("pixel_spacing") is None or len(ref["pixel_spacing"]) != 2:
        raise IrregularGeometryError(f"pid {pid}: missing PixelSpacing in {ref['path']}")
    for h in headers:
        ps = h.get("pixel_spacing")
        if ps is None or len(ps) != 2 or max(
                abs(a - b) for a, b in zip(ps, ref["pixel_spacing"])) > PIXEL_SPACING_TOL_MM:
            raise IrregularGeometryError(
                f"pid {pid}: mixed in-plane grid, PixelSpacing {ps} vs {ref['pixel_spacing']}")
        if h.get("matrix") != ref.get("matrix"):
            raise IrregularGeometryError(
                f"pid {pid}: mixed in-plane grid, matrix {h.get('matrix')} vs {ref.get('matrix')}")
        if max(abs(a - b) for a, b in zip(h["ipp_xy"], ref["ipp_xy"])) > IN_PLANE_ORIGIN_TOL_MM:
            raise IrregularGeometryError(
                f"pid {pid}: mixed in-plane grid, origin (x, y) {h['ipp_xy']} vs {ref['ipp_xy']}")

    steps = np.diff(sorted(h["z_position"] for h in headers))
    dz = float(np.median(steps))
    bad = steps[np.abs(steps - dz) > Z_STEP_TOL_MM]
    if bad.size:
        raise IrregularGeometryError(
            f"pid {pid}: irregular z steps {sorted(set(np.round(bad, 3).tolist()))} mm "
            f"(median {dz:g} mm)")

    thick = {h["slice_thickness"] for h in headers}
    if len(thick) != 1 or None in thick:
        raise IrregularGeometryError(f"pid {pid}: slice thickness not unique: {sorted(map(str, thick))}")
    t = thick.pop()
    if abs(dz - t) > CONTIGUITY_REL_TOL * t:
        raise IrregularGeometryError(
            f"pid {pid}: z step {dz:g} mm vs slice thickness {t:g} mm "
            f"(overlapping or gapped reconstruction)")
    return dz


@dataclass(frozen=True)
class SelectedSeries:
    """The D006-selected, D039-checked series of one patient, z ascending."""

    headers: tuple[dict, ...]
    dz: float

    @property
    def head(self) -> dict:
        return self.headers[0]

    @property
    def slice_positions(self) -> tuple[float, ...]:
        return tuple(m["z_position"] for m in self.headers)


def read_selected_headers(patient_dir: Path, pid: str) -> list[dict]:
    """Every header under ``patient_dir``, reduced to one series (D006).
    Unchecked and unsorted; :func:`select_series` is the normal entry point."""
    dcm_dir = find_dicom_subfolder(patient_dir)
    dcm_paths = [str(dcm_dir / f) for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files in {dcm_dir}")
    return _select_single_series([_read_header(p) for p in dcm_paths], pid)


def select_series(patient_dir: Path, pid: str) -> SelectedSeries:
    """Read every header under ``patient_dir``, select one series (D006) and
    check its geometry (D039). Raises :class:`IrregularGeometryError`."""
    headers = read_selected_headers(patient_dir, pid)
    dz = check_geometry(headers, pid)
    headers = sorted(headers, key=lambda m: (m["z_position"], m["instance_number"]))
    return SelectedSeries(headers=tuple(headers), dz=dz)


def load_patient_dicom(pid: str, data_root: Path) -> LoadedPatient:
    """Load a patient's CT series.

    Steps:
      1. :func:`select_series`: one series (D006), geometry checked (D039),
         sorted by (Z ascending, InstanceNumber tiebreak).
      2. Build the SimpleITK image from the chosen file list and confirm its
         z spacing equals the checked z step.
      3. Return :class:`LoadedPatient` with slice positions + scanner metadata.

    Parameters
    ----------
    pid : str
        Patient ID (directory name under ``data_root / "raw"``).
    data_root : Path
        Root path containing the ``raw/`` subdirectory (``paths.data_raw``'s
        parent; ``load_config`` enforces that layout).
    """
    sel = select_series(data_root / "raw" / str(pid), str(pid))
    sorted_files = [m["path"] for m in sel.headers]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    ct_sitk = reader.Execute()

    spacing = ct_sitk.GetSpacing()  # (x, y, z) mm
    if abs(spacing[2] - sel.dz) > Z_STEP_TOL_MM:
        raise IrregularGeometryError(
            f"pid {pid}: SimpleITK z spacing {spacing[2]:g} != checked z step {sel.dz:g}")

    head = sel.head
    return LoadedPatient(
        pid=str(pid),
        ct_sitk=ct_sitk,
        slice_positions=sel.slice_positions,
        pixel_spacing=spacing,
        slice_thickness=head["slice_thickness"],
        n_slices=len(sorted_files),
        scanner_model=head["scanner_model"],
        kernel=head["kernel"],
        manufacturer=head["manufacturer"],
        series_uid=head["series_uid"],
    )


@dataclass(frozen=True)
class PatientMetadata:
    """Header-only summary of a patient's DICOM series.

    Identical fields to :class:`LoadedPatient` except there's no ``ct_sitk``.
    Used by downstream stages (notably features) that already have the
    preprocessed CT on disk and only need the spatial metadata.
    """
    pid: str
    slice_positions: tuple[float, ...]
    pixel_spacing: tuple[float, float, float]
    slice_thickness: float
    n_slices: int
    scanner_model: str
    kernel: str
    manufacturer: str
    series_uid: str


def load_patient_metadata(pid: str, data_root: Path) -> PatientMetadata:
    """Header-only version of :func:`load_patient_dicom`.

    Same :func:`select_series` call, so the same files and the same checks.
    The z spacing is the checked z step (what SimpleITK assigns to the
    volume), not SliceThickness. ~10x faster because it skips the pixel read.
    """
    sel = select_series(data_root / "raw" / str(pid), str(pid))
    head = sel.head
    ds = pydicom.dcmread(head["path"], stop_before_pixels=True)
    row_sp = float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") else 0.0
    col_sp = float(ds.PixelSpacing[1]) if hasattr(ds, "PixelSpacing") else 0.0

    return PatientMetadata(
        pid=str(pid),
        slice_positions=sel.slice_positions,
        pixel_spacing=(col_sp, row_sp, sel.dz),
        slice_thickness=head["slice_thickness"],
        n_slices=len(sel.headers),
        scanner_model=head["scanner_model"],
        kernel=head["kernel"],
        manufacturer=head["manufacturer"],
        series_uid=head["series_uid"],
    )
