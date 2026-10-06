"""Build a 3D binary calcium mask from XML polygon annotations.

For each ROI in the parsed XML:

1. Find the matching CT slice index (Z-coordinate matching primary,
   ``ImageIndex`` fallback).
2. Rasterise the polygon (in pixel coordinates) onto that slice with
   ``cv2.fillPoly``.

Dirty-vessel ROIs (``vessel is None``) are skipped by default; this is
configurable via ``skip_dirty``. ROIs identified by the orchestration
script as failing the XML round-trip (D002) are excluded via
``excluded_roi_ids``: this keeps wrong-slice or otherwise unreliable
ROIs out of the final mask.

The builder returns the mask and a :class:`MaskBuildReport` that records
skipped/unmatched counts so the orchestration script can surface them
per patient.

Decisions referencing this module:
    D001: Z-coordinate matching for slice mapping
    D002: Round-trip-driven exclusion of unreliable ROIs
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import SimpleITK as sitk

from predict.io.dicom_loader import LoadedPatient
from predict.io.xml_parser import ParseResult, ROI
from predict.preprocess.slice_matcher import (
    Z_MATCH_TOLERANCE_MM,
    resolve_roi_slice,
)


@dataclass(frozen=True)
class MaskBuildReport:
    pid: str
    n_rois_total: int
    n_rasterised: int
    n_skipped_dirty: int
    n_skipped_too_few_points: int
    n_skipped_no_match: int
    n_matched_by_fallback: int
    n_excluded_by_roundtrip: int = 0


# Sub-pixel precision for polygon rasterisation. OsiriX/Horos store float
# Point_px vertices; their per-ROI Mean HU is computed against the float
# polygon, not an int-rounded one. We mirror that with cv2's ``shift``
# parameter (vertices scaled by 2^SHIFT, polygon edges drawn at 1/2^SHIFT
# pixel resolution). See D002.
_FILLPOLY_SHIFT: int = 4
_FILLPOLY_SCALE: int = 1 << _FILLPOLY_SHIFT


def _polygon_to_subpixel_points(roi: ROI) -> np.ndarray | None:
    """Convert an ROI's float ``points_px`` to a ``(N, 1, 2)`` int array
    scaled by ``2^_FILLPOLY_SHIFT`` for sub-pixel ``cv2.fillPoly``."""
    if roi.n_points < 3 or len(roi.points_px) < 3:
        return None
    pts = np.asarray(
        [[int(round(x * _FILLPOLY_SCALE)), int(round(y * _FILLPOLY_SCALE))]
         for x, y in roi.points_px],
        dtype=np.int32,
    )
    return pts.reshape((-1, 1, 2))


def build_3d_mask(
    parse_result: ParseResult,
    loaded: LoadedPatient,
    *,
    tolerance_mm: float = Z_MATCH_TOLERANCE_MM,
    skip_dirty: bool = True,
    excluded_roi_ids: set | None = None,
    vessel_filter: str | None = None,
) -> tuple[np.ndarray, MaskBuildReport]:
    """Build the 3D binary mask volume for a patient.

    Parameters
    ----------
    excluded_roi_ids
        Optional set of ``(image_index, roi_idx_in_slice)`` tuples to skip.
        Use :func:`predict.validate.xml_roundtrip.failed_roi_ids` to derive
        this set from a prior round-trip pass; the result is a mask cleaned
        of ROIs known to fail the slice or polygon check.
    vessel_filter
        If given (e.g. ``"LAD"``), only ROIs whose ``roi.vessel`` matches this
        canonical name are rasterised. Used by
        :mod:`predict.features.per_artery_mask` to build per-vessel masks
        without re-implementing the rasterisation. The exclusion set is still
        applied; ROI indices preserve their position in the unfiltered
        parse_result so ``excluded_roi_ids`` remains correct.

    Returns a ``(n_slices, height, width)`` uint8 array (mask) and a
    :class:`MaskBuildReport`. The mask is *not* wrapped in a SimpleITK image,
    callers that need spatial metadata should wrap it themselves and call
    ``CopyInformation(loaded.ct_sitk)``.
    """
    ct_arr = sitk.GetArrayFromImage(loaded.ct_sitk)
    n_slices, h, w = ct_arr.shape
    mask = np.zeros((n_slices, h, w), dtype=np.uint8)
    excluded = excluded_roi_ids or set()

    n_total = 0
    n_rast = 0
    n_dirty = 0
    n_few = 0
    n_unmatched = 0
    n_fallback = 0
    n_excluded = 0

    for slice_ann in parse_result.slices:
        for roi_idx, roi in enumerate(slice_ann.rois):
            n_total += 1

            if (slice_ann.image_index, roi_idx) in excluded:
                n_excluded += 1
                continue

            if skip_dirty and roi.vessel is None:
                n_dirty += 1
                continue

            if vessel_filter is not None and roi.vessel != vessel_filter:
                # Not counted under any skip bucket: out of scope for this call.
                continue

            poly = _polygon_to_subpixel_points(roi)
            if poly is None:
                n_few += 1
                continue

            idx, via = resolve_roi_slice(
                roi, slice_ann.image_index, loaded.slice_positions, tolerance_mm,
            )
            if via == "image_index_fallback":
                n_fallback += 1
            if idx is None:
                n_unmatched += 1
                continue

            cv2.fillPoly(mask[idx], [poly], color=1, shift=_FILLPOLY_SHIFT)
            n_rast += 1

    return mask, MaskBuildReport(
        pid=loaded.pid,
        n_rois_total=n_total,
        n_rasterised=n_rast,
        n_skipped_dirty=n_dirty,
        n_skipped_too_few_points=n_few,
        n_skipped_no_match=n_unmatched,
        n_matched_by_fallback=n_fallback,
        n_excluded_by_roundtrip=n_excluded,
    )


def mask_to_sitk(mask: np.ndarray, ct_sitk: sitk.Image) -> sitk.Image:
    """Wrap a mask array as a SimpleITK image with the CT's spatial metadata."""
    img = sitk.GetImageFromArray(mask)
    img.CopyInformation(ct_sitk)
    return img
