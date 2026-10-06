"""Per-vessel ROI counts in Agatston HU tiers + dense-calcium count.

The four tiers are the standard Agatston density-factor groups:

    d1: 130 ≤ HU < 200   (factor 1)
    d2: 200 ≤ HU < 300   (factor 2)
    d3: 300 ≤ HU < 400   (factor 3)
    d4: HU ≥ 400         (factor 4)

ROIs with ``max_hu < 130`` are below the calcium-detection threshold (D009)
and are never assigned to a tier. They simply don't contribute to any count.

``dense_calcium_count`` is the total number of ROIs anywhere in the patient
with ``max_hu > 1000`` (Hoori 2024 protective-signal threshold, D025).

Outputs (17 keys):

    n_rois_d{1..4}_{lad,rca,lcx,lm}              16
    dense_calcium_count                            1

Decisions referencing this module:
    D012: excluded_roi_ids as explicit input.
"""
from __future__ import annotations

from predict.config import VESSEL_NAMES
from predict.io.xml_parser import ParseResult


AGATSTON_HU_THRESHOLD: int = 130
DENSITY_TIER_EDGES: tuple[int, int, int] = (200, 300, 400)
DENSE_CALCIUM_HU_THRESHOLD: int = 1000
TIER_NAMES: tuple[str, ...] = ("d1", "d2", "d3", "d4")


def density_tier(max_hu: float) -> str | None:
    """Return the tier label for ``max_hu``, or None if below the Agatston threshold."""
    if max_hu < AGATSTON_HU_THRESHOLD:
        return None
    if max_hu < DENSITY_TIER_EDGES[0]:
        return "d1"
    if max_hu < DENSITY_TIER_EDGES[1]:
        return "d2"
    if max_hu < DENSITY_TIER_EDGES[2]:
        return "d3"
    return "d4"


def compute_density_tiers(
    parse_result: ParseResult,
    *,
    excluded_roi_ids: set | None = None,
) -> dict[str, float]:
    """Count ROIs per vessel per tier + dense-calcium total."""
    excluded = excluded_roi_ids or set()

    counts: dict[str, dict[str, int]] = {
        v: {t: 0 for t in TIER_NAMES} for v in VESSEL_NAMES
    }
    dense = 0

    for slice_ann in parse_result.slices:
        for roi_idx, roi in enumerate(slice_ann.rois):
            if roi.vessel is None:
                continue
            if (slice_ann.image_index, roi_idx) in excluded:
                continue

            tier = density_tier(roi.max_hu)
            if tier is not None:
                counts[roi.vessel][tier] += 1
            if roi.max_hu > DENSE_CALCIUM_HU_THRESHOLD:
                dense += 1

    out: dict[str, float] = {}
    for v in VESSEL_NAMES:
        suf = v.lower()
        for t in TIER_NAMES:
            out[f"n_rois_{t}_{suf}"] = float(counts[v][t])
    out["dense_calcium_count"] = float(dense)
    return out
