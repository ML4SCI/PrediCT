"""ROI-to-slice mapping by physical Z coordinate.

The XML's per-ROI ``Center`` field carries the lesion centroid in world
coordinates (x, y, z) in millimetres. Each DICOM slice's
``ImagePositionPatient[2]`` is its world Z. Matching ``Center[2]`` against the
slice Z list (provided by ``predict.io.dicom_loader.LoadedPatient``) gives an
unambiguous 0-based array index into the CT volume.

If the closest slice's Z is more than ``tolerance_mm`` away from the ROI
centre, the match is rejected (returns ``None``). This catches the patient
268 scenario where the DICOM series spacing no longer matches the XML's
original 1.5 mm reconstruction.

:func:`resolve_roi_slice` is the one rule every caller uses (round-trip,
mask builder, lesion grouping): the ``ImageIndex`` fallback applies only when
the ROI has no ``Center``. A ``Center`` that matches no slice is a
contradiction between the annotation and the scan, so the ROI is unmatched;
falling back to ``ImageIndex`` there would let it through on a guess.

Decisions referencing this module:
    D001: Z-coordinate matching as primary
"""
from __future__ import annotations

from typing import Sequence

from predict.io.xml_parser import ROI

# D001: a Center matches a slice only within half the native 3 mm slice
# step. The one value every ROI-to-slice caller uses (stage 2 round-trip and
# mask, stage 3 lesion grouping, D033 rebuilds), defined once so a change of
# the target z grid cannot split them.
Z_MATCH_TOLERANCE_MM: float = 1.5


def match_roi_to_slice(
    roi: ROI,
    slice_positions: Sequence[float],
    tolerance_mm: float = Z_MATCH_TOLERANCE_MM,
) -> int | None:
    """Return the 0-based array index of the slice matching ``roi.center_xyz[2]``.

    Returns ``None`` if the ROI lacks a centre or the closest slice is more
    than ``tolerance_mm`` away in Z.
    """
    if roi.center_xyz is None:
        return None
    z_roi = roi.center_xyz[2]

    best_idx = 0
    best_delta = abs(slice_positions[0] - z_roi)
    for i in range(1, len(slice_positions)):
        d = abs(slice_positions[i] - z_roi)
        if d < best_delta:
            best_delta = d
            best_idx = i

    if best_delta > tolerance_mm:
        return None
    return best_idx


def fallback_image_index_to_slice(image_index: int, n_slices: int) -> int | None:
    """Direct fallback when no ``Center`` is present.

    The convention is ``slice_idx = ImageIndex`` (no inversion). This is the
    correct mapping for the COCA XML format when DICOM files are sorted by
    Z ascending, as ``predict.io.dicom_loader.load_patient_dicom`` does.
    """
    if 0 <= image_index < n_slices:
        return image_index
    return None


def resolve_roi_slice(
    roi: ROI,
    image_index: int,
    slice_positions: Sequence[float],
    tolerance_mm: float = Z_MATCH_TOLERANCE_MM,
) -> tuple[int | None, str]:
    """Return ``(slice_idx, matched_via)`` for one ROI.

    ``matched_via`` is ``"center"`` (Z match), ``"image_index_fallback"``
    (no ``Center``, valid ``ImageIndex``), ``"center_mismatch"`` (``Center``
    present but no slice within ``tolerance_mm``; never falls back) or
    ``"unmatched"`` (no ``Center`` and ``ImageIndex`` out of range).
    """
    if roi.center_xyz is not None:
        idx = match_roi_to_slice(roi, slice_positions, tolerance_mm)
        return (idx, "center") if idx is not None else (None, "center_mismatch")
    idx = fallback_image_index_to_slice(image_index, len(slice_positions))
    return (idx, "image_index_fallback") if idx is not None else (None, "unmatched")
