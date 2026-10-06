"""Agatston scoring from XML annotations (D011 single-helper).

Agatston scoring follows the original 1990 specification:

    per-ROI score = Area_cm² × 100 × density_factor(max_HU) × (thickness / 3.0)
    patient score = Σ per-ROI score      (ROIs with max_HU < 130 contribute 0)

Density factor follows the standard 4-tier table:

    max_HU < 200    → 1
    200 ≤ HU < 300  → 2
    300 ≤ HU < 400  → 3
    max_HU ≥ 400    → 4

The thickness correction ``(slice_thickness_mm / 3.0)`` scales any non-3 mm
acquisition to the original 3 mm reference. On COCA's 3 mm cohort the factor
is 1.0; on a 2.5 mm cohort it would be 0.833; on 1.5 mm it would be 0.5.

This module exposes a single helper :func:`agatston_roi_score` used by both
:func:`compute_agatston` here and :func:`compute_per_vessel_aggregates`
in ``per_vessel_aggregates.py`` (D011: no duplicated formula).

Inputs / outputs and exclusions follow D012: the caller passes an
``excluded_roi_ids`` set of ``(image_index, roi_idx_in_slice)`` tuples that
will be skipped from the score (round-trip-failed ROIs are not eligible).

Dirty-vessel ROIs (``roi.vessel is None``) are silently skipped regardless of
the exclusion set.

Decisions referencing this module:
    D011: Single Agatston thickness-correction helper.
    D012: excluded_roi_ids as explicit input.
"""
from __future__ import annotations

from dataclasses import dataclass

from predict.config import VESSEL_NAMES
from predict.io.xml_parser import ParseResult


AGATSTON_HU_THRESHOLD: int = 130
AGATSTON_AREA_FACTOR: float = 100.0       # cm² → mm² conversion for the standard formula
REFERENCE_THICKNESS_MM: float = 3.0


def density_factor(max_hu: float) -> int:
    """Return the Agatston density-weight factor for a peak HU value.

    HU below 130 is the calcium-detection threshold and is normally caller-
    filtered; this function never returns 0, but the caller must check the
    threshold first to know whether to include the ROI at all.
    """
    if max_hu < 200:
        return 1
    if max_hu < 300:
        return 2
    if max_hu < 400:
        return 3
    return 4


def agatston_roi_score(
    area_cm2: float,
    max_hu: float,
    slice_thickness_mm: float,
) -> float:
    """Per-ROI Agatston score (D011 single helper).

    Returns 0.0 if ``max_hu < AGATSTON_HU_THRESHOLD``. Negative inputs are
    treated as 0 and produce 0.0 (defensive).
    """
    if max_hu < AGATSTON_HU_THRESHOLD or area_cm2 <= 0 or slice_thickness_mm <= 0:
        return 0.0
    return (
        area_cm2
        * AGATSTON_AREA_FACTOR
        * density_factor(max_hu)
        * (slice_thickness_mm / REFERENCE_THICKNESS_MM)
    )


def classify_risk(total: float) -> str:
    """AHA / CAC-DRS risk category from total Agatston score."""
    if total <= 0:
        return "0"
    if total < 100:
        return "1-99"
    if total < 400:
        return "100-399"
    return "400+"


@dataclass(frozen=True)
class AgatstonResult:
    """Per-patient Agatston output."""

    total: float
    per_vessel: dict[str, float]   # keys are canonical names: LAD, RCA, LCx, LM
    category: str

    def to_feature_dict(self) -> dict[str, float]:
        """Flatten to the feature-schema keys: ``agatston_{suf}`` + ``agatston_total``."""
        out: dict[str, float] = {}
        for vessel, score in self.per_vessel.items():
            out[f"agatston_{vessel.lower()}"] = float(score)
        out["agatston_total"] = float(self.total)
        return out


def compute_agatston(
    parse_result: ParseResult,
    *,
    slice_thickness_mm: float,
    excluded_roi_ids: set | None = None,
) -> AgatstonResult:
    """Compute per-vessel and total Agatston from XML annotations.

    Parameters
    ----------
    parse_result
        Output of :func:`predict.io.xml_parser.parse_calcium_xml`.
    slice_thickness_mm
        Native CT slice thickness in mm.
    excluded_roi_ids
        Optional set of ``(image_index, roi_idx_in_slice)`` tuples to skip.
    """
    excluded = excluded_roi_ids or set()
    per_vessel: dict[str, float] = {v: 0.0 for v in VESSEL_NAMES}

    for slice_ann in parse_result.slices:
        for roi_idx, roi in enumerate(slice_ann.rois):
            if roi.vessel is None:
                continue
            if (slice_ann.image_index, roi_idx) in excluded:
                continue
            per_vessel[roi.vessel] += agatston_roi_score(
                area_cm2=roi.area_cm2,
                max_hu=roi.max_hu,
                slice_thickness_mm=slice_thickness_mm,
            )

    total = sum(per_vessel.values())
    return AgatstonResult(
        total=total,
        per_vessel=per_vessel,
        category=classify_risk(total),
    )
