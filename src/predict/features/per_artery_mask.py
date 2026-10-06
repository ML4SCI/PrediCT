"""Build a separate 3D binary mask per canonical vessel (D008).

For each vessel in ``LAD``, ``RCA``, ``LCx``, ``LM``, call the same
``mask_builder.build_3d_mask`` used for the whole-mask in stage 2, with a
``vessel_filter`` that restricts rasterisation to that vessel. The same
``excluded_roi_ids`` is applied: per-vessel masks are voxel-consistent with
the whole-mask: summing the four per-vessel voxel counts equals the whole-
mask voxel count (modulo zero overlap, which is guaranteed by the disjoint
vessel labels).

This is the input layer for per-vessel PyRadiomics and per-vessel
shape/intensity features. The XML-derived per-vessel scalars (volume, mass,
mean/max HU, Agatston, density tiers) come from the dedicated modules and
do not need a mask.

Decisions referencing this module:
    D008: Per-artery masks via filter-then-rasterise.
    D012: excluded_roi_ids consistency with whole-mask.
"""
from __future__ import annotations

import numpy as np

from predict.config import VESSEL_NAMES
from predict.io.dicom_loader import LoadedPatient
from predict.io.xml_parser import ParseResult
from predict.preprocess.mask_builder import build_3d_mask
from predict.preprocess.slice_matcher import Z_MATCH_TOLERANCE_MM


def build_per_artery_masks(
    parse_result: ParseResult,
    loaded: LoadedPatient,
    *,
    excluded_roi_ids: set | None = None,
    tolerance_mm: float = Z_MATCH_TOLERANCE_MM,
) -> dict[str, np.ndarray]:
    """Return one ``(n_slices, H, W)`` uint8 mask per canonical vessel.

    Empty vessels (no rasterisable ROIs after exclusions) yield an all-zero
    mask of the correct shape.
    """
    masks: dict[str, np.ndarray] = {}
    for vessel in VESSEL_NAMES:
        mask, _ = build_3d_mask(
            parse_result,
            loaded,
            tolerance_mm=tolerance_mm,
            excluded_roi_ids=excluded_roi_ids,
            vessel_filter=vessel,
        )
        masks[vessel] = mask
    return masks


def per_artery_voxel_counts(masks: dict[str, np.ndarray]) -> dict[str, int]:
    """Convenience: voxel count per vessel (sanity-check + reporting)."""
    return {v: int(m.sum()) for v, m in masks.items()}
