"""Per-vessel scalar features derived directly from XML annotations.

This module covers the per-vessel volume / mass / HU stats and their patient-
level totals. It is XML-only (no mask, no CT) and very cheap to run.

Mass formula (D015: HU-volume product, no calibration):

    mass_vessel = Σ_ROI (area_mm² × slice_thickness_mm × mean_HU)

Volume:

    volume_vessel = Σ_ROI (area_mm² × slice_thickness_mm)

These are uncalibrated within-cohort quantities, equivalent to the calibrated
clinical mass score up to a scanner phantom constant which is absent from
COCA metadata. The constant cancels in any ranking / clustering / distance
operation, so the omission is harmless for phenotyping.

Per-vessel HU summaries:

    max_hu_vessel  = max over ROIs of roi.max_hu        (0 if no ROIs)
    mean_hu_vessel = Σ(area_mm² × mean_hu) / Σ area     (area-weighted)

Patient-level totals are simple sums over the four vessels. The global
mean-HU is also area-weighted across all ROIs.

Agatston per vessel comes from :mod:`predict.features.agatston`, not from
this module: see D011 (one helper, one call site).

Decisions referencing this module:
    D011: Agatston helper lives in agatston.py (not duplicated here).
    D012: excluded_roi_ids skips round-trip-failed ROIs.
"""
from __future__ import annotations

from predict.config import VESSEL_NAMES
from predict.io.xml_parser import ParseResult


def compute_per_vessel_aggregates(
    parse_result: ParseResult,
    *,
    slice_thickness_mm: float,
    excluded_roi_ids: set | None = None,
) -> dict[str, float]:
    """Return per-vessel volume / mass / HU stats plus global totals.

    Output keys (16 per-vessel + 4 globals = 20):

      volume_{lad,rca,lcx,lm}_mm3
      mass_{lad,rca,lcx,lm}
      mean_hu_{lad,rca,lcx,lm}
      max_hu_{lad,rca,lcx,lm}
      volume_total_mm3
      mass_total
      mean_hu_weighted_global
      max_hu_global
    """
    excluded = excluded_roi_ids or set()

    # Per-vessel accumulators (canonical names: LAD, RCA, LCx, LM).
    area_sum: dict[str, float] = {v: 0.0 for v in VESSEL_NAMES}
    mass_sum: dict[str, float] = {v: 0.0 for v in VESSEL_NAMES}
    weighted_hu_num: dict[str, float] = {v: 0.0 for v in VESSEL_NAMES}
    max_hu: dict[str, float] = {v: 0.0 for v in VESSEL_NAMES}

    for slice_ann in parse_result.slices:
        for roi_idx, roi in enumerate(slice_ann.rois):
            if roi.vessel is None:
                continue
            if (slice_ann.image_index, roi_idx) in excluded:
                continue

            v = roi.vessel
            area_mm2 = roi.area_cm2 * 100.0
            area_sum[v] += area_mm2
            mass_sum[v] += area_mm2 * slice_thickness_mm * roi.mean_hu
            weighted_hu_num[v] += area_mm2 * roi.mean_hu
            if roi.max_hu > max_hu[v]:
                max_hu[v] = roi.max_hu

    out: dict[str, float] = {}
    total_area = 0.0
    total_mass = 0.0
    total_weighted_hu_num = 0.0
    overall_max = 0.0

    for v in VESSEL_NAMES:
        suf = v.lower()
        volume = area_sum[v] * slice_thickness_mm
        mass = mass_sum[v]
        mean_hu = weighted_hu_num[v] / area_sum[v] if area_sum[v] > 0 else 0.0
        out[f"volume_{suf}_mm3"] = volume
        out[f"mass_{suf}"] = mass
        out[f"mean_hu_{suf}"] = mean_hu
        out[f"max_hu_{suf}"] = max_hu[v]

        total_area += area_sum[v]
        total_mass += mass
        total_weighted_hu_num += weighted_hu_num[v]
        if max_hu[v] > overall_max:
            overall_max = max_hu[v]

    out["volume_total_mm3"] = total_area * slice_thickness_mm
    out["mass_total"] = total_mass
    out["mean_hu_weighted_global"] = (
        total_weighted_hu_num / total_area if total_area > 0 else 0.0
    )
    out["max_hu_global"] = overall_max

    return out
