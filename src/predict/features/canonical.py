"""The 68 canonical (XML-derived) features for one patient, in one call.

Stage 3 and the D033 realistic-perturbation stage both need the same
canonical features: stage 3 from the parsed XML, D033 from a boundary-jittered
copy of it. One function keeps them from drifting apart
(directive: one rule, one implementation).

None of this reads the CT array (D016). Inputs are the parsed XML, the
patient's native DICOM metadata and the D002 exclusion set.

Decisions referencing this module:
    D007 - Lesion grouping (via group_rois_into_lesions)
    D011 - Agatston thickness handling (via compute_agatston)
    D016 - Canonical features never read the CT
    D033 - Recomputed per draw from jittered polygons
"""
from __future__ import annotations

from typing import Any

from predict.features.agatston import compute_agatston
from predict.features.density_tiers import compute_density_tiers
from predict.features.feature_schema import zero_features
from predict.features.lesion_ccl import group_rois_into_lesions
from predict.features.per_vessel_aggregates import compute_per_vessel_aggregates
from predict.features.spatial import compute_spatial_features
from predict.io.xml_parser import ParseResult


def compute_canonical_features(
    parse_result: ParseResult,
    meta: Any,
    excluded_roi_ids: set,
    *,
    max_inplane_mm: float,
    max_slice_gap: int,
) -> tuple[dict[str, float], str, dict]:
    """Return ``(features, agatston_category, lesions_per_vessel)``.

    ``max_inplane_mm`` and ``max_slice_gap`` are the D007 lesion-grouping
    parameters. They are required, with no defaults, so every caller passes
    the configured ``lesion_grouping`` values (:func:`grouping_from_config`)
    and a config change cannot be silently ignored.

    ``features`` starts from ``zero_features()`` and is updated in the same
    order stage 3 always used, so it holds the same keys. ``meta`` is a
    :class:`predict.io.PatientMetadata` (or anything with ``slice_positions``,
    ``pixel_spacing`` and ``slice_thickness``).
    """
    feats = zero_features()

    ag = compute_agatston(
        parse_result,
        slice_thickness_mm=meta.slice_thickness,
        excluded_roi_ids=excluded_roi_ids,
    )
    feats.update(ag.to_feature_dict())

    feats.update(compute_per_vessel_aggregates(
        parse_result,
        slice_thickness_mm=meta.slice_thickness,
        excluded_roi_ids=excluded_roi_ids,
    ))
    feats.update(compute_density_tiers(parse_result, excluded_roi_ids=excluded_roi_ids))

    lesions_per_vessel = group_rois_into_lesions(
        parse_result,
        slice_positions=meta.slice_positions,
        pixel_spacing_xy=(meta.pixel_spacing[0], meta.pixel_spacing[1]),
        slice_thickness_mm=meta.slice_thickness,
        excluded_roi_ids=excluded_roi_ids,
        max_inplane_mm=max_inplane_mm,
        max_slice_gap=max_slice_gap,
    )
    feats.update(compute_spatial_features(
        lesions_per_vessel,
        slice_positions=meta.slice_positions,
        slice_thickness_mm=meta.slice_thickness,
    ))
    return feats, ag.category, lesions_per_vessel


def grouping_from_config(cfg: Any) -> dict:
    """The D007 keyword arguments for :func:`compute_canonical_features`."""
    return {
        "max_inplane_mm": float(cfg.lesion_grouping.max_inplane_dist_mm),
        "max_slice_gap": int(cfg.lesion_grouping.max_slice_gap),
    }


__all__ = ["compute_canonical_features", "grouping_from_config"]
