"""3D lesion identification by per-vessel BFS connected components (D007).

A "lesion" here is a maximal set of XML ROIs that:

  - share the same canonical vessel (``LAD`` / ``RCA`` / ``LCx`` / ``LM``);
  - have CT array slice indices within ``max_slice_gap`` (default 1);
  - have in-plane (XY) centroid distance ≤ ``max_inplane_mm`` (default 5 mm);
  - are eligible: neither dirty (``vessel is None``) nor in the caller's
    ``excluded_roi_ids`` set (D012);
  - have a slice index from ``resolve_roi_slice`` (Z match; the ImageIndex
    fallback only when no ``Center`` is present, the same rule as
    mask_builder and the D002 round-trip).

Pipeline callers pass ``max_inplane_mm`` / ``max_slice_gap`` from the config
(``predict.features.canonical.grouping_from_config``); the defaults here are
for direct use and tests. Linking by slice INDEX means millimetres only
because D039 guarantees evenly spaced, contiguous native slices.

Each lesion is summarised as a :class:`Lesion` dataclass carrying the keys of
its constituent ROIs (so callers can map back), the area-weighted physical
centroid, total area / volume, max HU, and area-weighted mean HU. Downstream
modules (``spatial``, ``per_vessel_aggregates``) consume these lesions to
produce patient-level features.

Decisions referencing this module:
    D007: Lesion grouping rule.
    D012: excluded_roi_ids as explicit input.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from predict.io.xml_parser import ParseResult, ROI
from predict.preprocess.slice_matcher import (
    Z_MATCH_TOLERANCE_MM,
    resolve_roi_slice,
)


# ───────────────────── Public dataclass ─────────────────────


@dataclass(frozen=True)
class Lesion:
    """A 3D connected component of XML ROIs in one vessel."""

    vessel: str
    roi_keys: tuple[tuple[int, int], ...]       # sorted (image_index, roi_idx_in_slice)
    slice_indices: tuple[int, ...]              # sorted unique CT array slice indices
    centroid_mm: tuple[float, float, float]     # (x, y, z) area-weighted, mm
    total_area_mm2: float
    mean_hu_weighted: float
    max_hu: float
    volume_mm3: float

    @property
    def n_rois(self) -> int:
        return len(self.roi_keys)


# ───────────────────── Internal node ─────────────────────


@dataclass(frozen=True)
class _Node:
    """One eligible ROI prepared for the BFS graph."""
    vessel: str
    image_index: int
    roi_idx: int
    slice_idx: int                  # CT array slice index, Z-matched (with fallback)
    centroid_xy_mm: tuple[float, float]
    z_mm: float                     # slice_positions[slice_idx]
    area_mm2: float
    mean_hu: float
    max_hu: float


# ───────────────────── Helpers ─────────────────────


def _polygon_centroid_px(points_px: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Area centroid of a polygon in (col, row) pixel coords (float).

    Uses ``cv2.moments`` for the true geometric centroid. Falls back to the
    arithmetic vertex mean if the polygon has zero algebraic area (degenerate;
    in practice never reached because the parser drops < 3-point ROIs).
    """
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    m = cv2.moments(pts)
    if m["m00"] != 0:
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])
    arr = np.asarray(points_px, dtype=float)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


def _build_nodes(
    parse_result: ParseResult,
    slice_positions: Sequence[float],
    pixel_spacing_xy: tuple[float, float],
    excluded: set,
    tolerance_mm: float,
) -> list[_Node]:
    """Turn eligible ROIs into ``_Node`` objects ready for BFS."""
    nodes: list[_Node] = []

    for slice_ann in parse_result.slices:
        for roi_idx, roi in enumerate(slice_ann.rois):
            if roi.vessel is None:
                continue
            key = (slice_ann.image_index, roi_idx)
            if key in excluded:
                continue
            if roi.n_points < 3 or len(roi.points_px) < 3:
                continue

            slice_idx, _ = resolve_roi_slice(
                roi, slice_ann.image_index, slice_positions, tolerance_mm,
            )
            if slice_idx is None:
                continue

            cx, cy = _polygon_centroid_px(roi.points_px)
            x_mm = cx * pixel_spacing_xy[0]
            y_mm = cy * pixel_spacing_xy[1]
            z_mm = float(slice_positions[slice_idx])

            nodes.append(_Node(
                vessel=roi.vessel,
                image_index=slice_ann.image_index,
                roi_idx=roi_idx,
                slice_idx=slice_idx,
                centroid_xy_mm=(x_mm, y_mm),
                z_mm=z_mm,
                area_mm2=roi.area_cm2 * 100.0,
                mean_hu=roi.mean_hu,
                max_hu=roi.max_hu,
            ))

    return nodes


def _bfs_components(
    nodes: list[_Node],
    max_inplane_mm: float,
    max_slice_gap: int,
) -> list[list[int]]:
    """Return connected components as lists of node indices.

    Edge predicate:
      |slice_idx_i - slice_idx_j| <= max_slice_gap
        AND  ||xy_i - xy_j||_2 <= max_inplane_mm
    """
    n = len(nodes)
    adj: list[list[int]] = [[] for _ in range(n)]

    inplane2 = max_inplane_mm * max_inplane_mm
    for i in range(n):
        ni = nodes[i]
        for j in range(i + 1, n):
            nj = nodes[j]
            if abs(ni.slice_idx - nj.slice_idx) > max_slice_gap:
                continue
            dx = ni.centroid_xy_mm[0] - nj.centroid_xy_mm[0]
            dy = ni.centroid_xy_mm[1] - nj.centroid_xy_mm[1]
            if dx * dx + dy * dy <= inplane2:
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        comp: list[int] = []
        q: deque = deque([start])
        seen[start] = True
        while q:
            k = q.popleft()
            comp.append(k)
            for m in adj[k]:
                if not seen[m]:
                    seen[m] = True
                    q.append(m)
        components.append(comp)
    return components


def _build_lesion(
    nodes: list[_Node],
    component_indices: list[int],
    slice_thickness_mm: float,
) -> Lesion:
    """Aggregate one connected component of nodes into a :class:`Lesion`."""
    members = [nodes[i] for i in component_indices]
    total_area = sum(m.area_mm2 for m in members)

    # Area-weighted centroid (XY in mm using each ROI's centroid; Z using each
    # ROI's slice z). Falls back to unweighted mean if all areas are zero,
    # which is a defensive guard: area=0 ROIs are filtered upstream.
    if total_area > 0:
        cx = sum(m.area_mm2 * m.centroid_xy_mm[0] for m in members) / total_area
        cy = sum(m.area_mm2 * m.centroid_xy_mm[1] for m in members) / total_area
        cz = sum(m.area_mm2 * m.z_mm for m in members) / total_area
        mean_hu_weighted = sum(m.area_mm2 * m.mean_hu for m in members) / total_area
    else:
        cx = sum(m.centroid_xy_mm[0] for m in members) / len(members)
        cy = sum(m.centroid_xy_mm[1] for m in members) / len(members)
        cz = sum(m.z_mm for m in members) / len(members)
        mean_hu_weighted = sum(m.mean_hu for m in members) / len(members)

    roi_keys = tuple(sorted((m.image_index, m.roi_idx) for m in members))
    slice_indices = tuple(sorted({m.slice_idx for m in members}))
    max_hu = max(m.max_hu for m in members)
    volume_mm3 = total_area * slice_thickness_mm

    return Lesion(
        vessel=members[0].vessel,
        roi_keys=roi_keys,
        slice_indices=slice_indices,
        centroid_mm=(cx, cy, cz),
        total_area_mm2=total_area,
        mean_hu_weighted=mean_hu_weighted,
        max_hu=max_hu,
        volume_mm3=volume_mm3,
    )


# ───────────────────── Public entry point ─────────────────────


def group_rois_into_lesions(
    parse_result: ParseResult,
    *,
    slice_positions: Sequence[float],
    pixel_spacing_xy: tuple[float, float],
    slice_thickness_mm: float,
    excluded_roi_ids: set | None = None,
    max_inplane_mm: float = 5.0,
    max_slice_gap: int = 1,
    tolerance_mm: float = Z_MATCH_TOLERANCE_MM,
) -> dict[str, list[Lesion]]:
    """Group eligible ROIs into 3D lesions per canonical vessel.

    Returns a dict keyed by canonical vessel name (``LAD``, ``RCA``, ``LCx``,
    ``LM``); the value is a list of :class:`Lesion` objects (possibly empty).
    Vessels with no eligible ROIs are still keys in the result with an empty
    list, so downstream code can iterate ``VESSEL_NAMES`` without KeyError.
    """
    excluded = excluded_roi_ids or set()
    nodes = _build_nodes(
        parse_result, slice_positions, pixel_spacing_xy, excluded, tolerance_mm,
    )

    by_vessel: dict[str, list[_Node]] = defaultdict(list)
    for node in nodes:
        by_vessel[node.vessel].append(node)

    out: dict[str, list[Lesion]] = {"LAD": [], "RCA": [], "LCx": [], "LM": []}
    for vessel, vnodes in by_vessel.items():
        components = _bfs_components(vnodes, max_inplane_mm, max_slice_gap)
        out[vessel] = [
            _build_lesion(vnodes, comp, slice_thickness_mm)
            for comp in components
        ]
        # Sort lesions deterministically: ascending z, then x, then y, then n_rois desc.
        out[vessel].sort(
            key=lambda l: (l.centroid_mm[2], l.centroid_mm[0], l.centroid_mm[1], -l.n_rois)
        )
    return out
