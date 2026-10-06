"""XML stat round-trip: preprocessing correctness gate.

For every non-dirty ROI in a parsed XML, this module:

1. Re-runs slice matching (Z-coordinate primary, ImageIndex fallback) to
   find the CT slice the ROI sits on.
2. Re-rasterises the polygon onto a fresh 2D mask.
3. Computes Mean / Max / Min HU from the CT voxels inside that mask.
4. Compares against the XML-stored ``Mean``, ``Max``, ``Min``.

Pass criterion (D002):

- **Hard gate**: ``voxel_max == xml_max`` exactly. This is the slice- and
  polygon-correctness signal: the brightest pixel inside our polygon on
  the matched slice is the same one OsiriX saw.
- **Informational**: ``delta_mean`` and ``delta_min`` are reported but do
  not fail the ROI unless ``|delta_mean| > 200 HU`` (gross mismatch). The
  binary mask we build is systematically slightly larger than OsiriX's
  partial-volume-weighted polygon, so a 20–60 HU mean drop is expected
  and constant across patients. See D002 for the full rationale.

This module is independent of :mod:`predict.preprocess.mask_builder`: it
re-derives the per-ROI 2D mask itself so a bug in mask_builder cannot mask
itself in the round-trip.

Decisions referencing this module:
    D002: XML stat round-trip as preprocessing correctness gate
    D001: Z-coordinate matching (the thing being verified)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import cv2
import numpy as np

from predict.io.xml_parser import ParseResult, ROI
from predict.preprocess.slice_matcher import Z_MATCH_TOLERANCE_MM, resolve_roi_slice


@dataclass(frozen=True)
class ROITrip:
    pid: str
    image_index: int
    roi_idx_in_slice: int     # 0-based index inside parse_result.slices[k].rois
    vessel: str | None
    matched_slice_idx: int | None
    matched_via: str  # "center" | "image_index_fallback" | "center_mismatch" | "unmatched" | "dirty" | "skipped"
    n_xml_points: int
    xml_mean: float
    xml_max: float
    xml_min: float
    voxel_n: int
    voxel_mean: float
    voxel_max: float
    voxel_min: float
    delta_mean: float
    delta_max: float
    delta_min: float
    passes: bool
    reason: str


# Sub-pixel polygon rasterisation; see mask_builder._FILLPOLY_SHIFT.
_FILLPOLY_SHIFT: int = 4
_FILLPOLY_SCALE: int = 1 << _FILLPOLY_SHIFT


def _rasterise_roi_2d(roi: ROI, shape: tuple[int, int]) -> np.ndarray | None:
    if roi.n_points < 3 or len(roi.points_px) < 3:
        return None
    pts = np.asarray(
        [[int(round(x * _FILLPOLY_SCALE)), int(round(y * _FILLPOLY_SCALE))]
         for x, y in roi.points_px],
        dtype=np.int32,
    ).reshape((-1, 1, 2))
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1, shift=_FILLPOLY_SHIFT)
    return mask


def check_roi(
    pid: str,
    image_index: int,
    roi: ROI,
    ct_array: np.ndarray,
    slice_positions: Sequence[float],
    *,
    roi_idx_in_slice: int = 0,
    tolerance_mm: float = Z_MATCH_TOLERANCE_MM,
    max_max_delta: float = 0.0,        # hard gate: voxel_max must match xml_max exactly
    max_mean_delta: float = 200.0,     # informational; only fails on gross mismatch
    include_dirty: bool = False,
) -> ROITrip:
    """Run the round-trip on a single ROI and return a :class:`ROITrip`."""
    n_slices, h, w = ct_array.shape

    def _empty(reason: str, matched_via: str, idx: int | None) -> ROITrip:
        return ROITrip(
            pid=pid,
            image_index=image_index,
            roi_idx_in_slice=roi_idx_in_slice,
            vessel=roi.vessel,
            matched_slice_idx=idx,
            matched_via=matched_via,
            n_xml_points=roi.n_points,
            xml_mean=roi.mean_hu,
            xml_max=roi.max_hu,
            xml_min=roi.min_hu,
            voxel_n=0,
            voxel_mean=float("nan"),
            voxel_max=float("nan"),
            voxel_min=float("nan"),
            delta_mean=float("nan"),
            delta_max=float("nan"),
            delta_min=float("nan"),
            passes=False,
            reason=reason,
        )

    if roi.vessel is None and not include_dirty:
        return _empty("dirty_vessel_skipped", "dirty", None)

    idx, matched_via = resolve_roi_slice(roi, image_index, slice_positions, tolerance_mm)
    if idx is None:
        reason = "center_outside_scan" if matched_via == "center_mismatch" else "no_slice_match"
        return _empty(reason, matched_via, None)

    mask2d = _rasterise_roi_2d(roi, (h, w))
    if mask2d is None or mask2d.sum() == 0:
        return _empty("polygon_invalid", matched_via, idx)

    voxels = ct_array[idx][mask2d == 1]
    v_n = int(voxels.size)
    v_mean = float(voxels.mean())
    v_max = float(voxels.max())
    v_min = float(voxels.min())

    d_mean = v_mean - roi.mean_hu
    d_max = v_max - roi.max_hu
    d_min = v_min - roi.min_hu

    # Hard gate: Max HU must match exactly (D002). Mean is informational
    # unless it exceeds the gross-mismatch threshold.
    passes = abs(d_max) <= max_max_delta and abs(d_mean) <= max_mean_delta
    if passes:
        reason = "ok"
    elif abs(d_max) > max_max_delta:
        reason = f"max_delta={d_max:+.2f}_exceeds_{max_max_delta:.2f}"
    else:
        reason = f"mean_delta={d_mean:+.2f}_exceeds_{max_mean_delta:.2f}"

    return ROITrip(
        pid=pid,
        image_index=image_index,
        roi_idx_in_slice=roi_idx_in_slice,
        vessel=roi.vessel,
        matched_slice_idx=idx,
        matched_via=matched_via,
        n_xml_points=roi.n_points,
        xml_mean=roi.mean_hu,
        xml_max=roi.max_hu,
        xml_min=roi.min_hu,
        voxel_n=v_n,
        voxel_mean=v_mean,
        voxel_max=v_max,
        voxel_min=v_min,
        delta_mean=d_mean,
        delta_max=d_max,
        delta_min=d_min,
        passes=passes,
        reason=reason,
    )


def xml_roundtrip_check(
    parse_result: ParseResult,
    ct_array: np.ndarray,
    slice_positions: Sequence[float],
    *,
    tolerance_mm: float = Z_MATCH_TOLERANCE_MM,
    max_max_delta: float = 0.0,
    max_mean_delta: float = 200.0,
    include_dirty: bool = False,
) -> list[ROITrip]:
    """Run the round-trip on every ROI in ``parse_result``."""
    trips: list[ROITrip] = []
    for slice_ann in parse_result.slices:
        for roi_idx, roi in enumerate(slice_ann.rois):
            trips.append(check_roi(
                pid=parse_result.pid,
                image_index=slice_ann.image_index,
                roi=roi,
                ct_array=ct_array,
                slice_positions=slice_positions,
                roi_idx_in_slice=roi_idx,
                tolerance_mm=tolerance_mm,
                max_max_delta=max_max_delta,
                max_mean_delta=max_mean_delta,
                include_dirty=include_dirty,
            ))
    return trips


def failed_roi_ids(trips: Sequence[ROITrip]) -> set:
    """Return the set of ``(image_index, roi_idx_in_slice)`` tuples for ROIs
    that should be excluded from the mask: non-dirty trips that did not pass
    the hard gate."""
    return {
        (t.image_index, t.roi_idx_in_slice)
        for t in trips
        if (not t.passes) and t.matched_via != "dirty"
    }


def pass_rate(trips: Sequence[ROITrip]) -> float:
    """Fraction of ROIs that passed, excluding dirty/skipped ones."""
    eligible = [t for t in trips if t.matched_via != "dirty"]
    if not eligible:
        return 1.0
    return sum(t.passes for t in eligible) / len(eligible)


def trips_to_rows(trips: Sequence[ROITrip]) -> list[dict]:
    """Flatten to dicts suitable for a DataFrame / CSV writer."""
    return [asdict(t) for t in trips]
