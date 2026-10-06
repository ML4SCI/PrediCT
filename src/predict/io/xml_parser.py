"""Parse COCA Apple plist XML calcium annotations.

Each XML file (one per patient) contains an ``Images`` array. Each entry has
an ``ImageIndex`` and a list of ``ROIs``. Each ROI has a vessel name, polygon
points in pixel and world coordinates, a centroid (``Center``), and pre-
computed HU statistics (``Mean``, ``Max``, ``Min``, ``Total``, ``Area``).

This parser exposes those fields verbatim, with two additions:

- Vessel names are normalised against ``VESSEL_NAME_MAP``. Names that do not
  match a known mapping are returned with ``vessel=None`` and reported in the
  parse summary as dirty.
- ROIs with ``NumberOfPoints == 0`` (placeholder entries) are dropped.

The ``Center`` field is the load-bearing primitive for D001 (Z-coordinate
matching).

Decisions referencing this module:
    D001: Center field is the source for Z-matching
"""
from __future__ import annotations

import plistlib
from dataclasses import dataclass, field
from pathlib import Path

from predict.config import VESSEL_NAME_MAP


@dataclass(frozen=True)
class ROI:
    vessel_raw: str                                       # exact string from XML
    vessel: str | None                                    # LAD/RCA/LCx/LM or None
    area_cm2: float                                       # Area in cm^2
    mean_hu: float
    max_hu: float
    min_hu: float
    total_hu: float
    n_points: int
    points_px: tuple[tuple[float, float], ...]            # (col, row), float
    points_mm: tuple[tuple[float, float, float], ...]     # (x, y, z), world
    center_xyz: tuple[float, float, float] | None         # 3D centroid, may be None


@dataclass(frozen=True)
class SliceAnnotation:
    image_index: int
    rois: tuple[ROI, ...]


@dataclass(frozen=True)
class ParseResult:
    pid: str
    slices: tuple[SliceAnnotation, ...]
    dirty_vessel_names: tuple[str, ...] = field(default_factory=tuple)
    n_active_rois: int = 0
    n_dropped_zero_point: int = 0


def _parse_point(value) -> tuple[float, ...]:
    """Parse a string like ``"(123.5, 47.2)"`` or ``"(1.0, 2.0, 3.0)"``."""
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    inner = str(value).strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    return tuple(float(p.strip()) for p in inner.split(","))


def _parse_center(value) -> tuple[float, float, float] | None:
    """Parse the ``Center`` field. Returns ``None`` if absent/malformed."""
    if value is None:
        return None
    try:
        parts = _parse_point(value)
    except (ValueError, AttributeError):
        return None
    if len(parts) != 3:
        return None
    return (parts[0], parts[1], parts[2])


def parse_calcium_xml(pid: str, xml_dir: Path) -> ParseResult:
    """Parse one patient's calcium XML.

    Parameters
    ----------
    pid : str
        Patient ID. The file is expected at ``xml_dir / f"{pid}.xml"``.
    xml_dir : Path
        Directory containing the per-patient XML files.
    """
    xml_path = xml_dir / f"{pid}.xml"
    with open(xml_path, "rb") as f:
        plist = plistlib.load(f)

    dirty: set[str] = set()
    n_active = 0
    n_dropped = 0
    out_slices: list[SliceAnnotation] = []

    for image_entry in plist.get("Images", []):
        out_rois: list[ROI] = []
        for r in image_entry.get("ROIs", []):
            n_pts = int(r.get("NumberOfPoints", 0))
            if n_pts == 0:
                n_dropped += 1
                continue

            raw_name = str(r.get("Name", "Unknown"))
            vessel = VESSEL_NAME_MAP.get(raw_name)
            if vessel is None:
                dirty.add(raw_name)

            pts_px = tuple(_parse_point(p)[:2] for p in r.get("Point_px", []))
            pts_mm = tuple(_parse_point(p)[:3] for p in r.get("Point_mm", []))

            out_rois.append(ROI(
                vessel_raw=raw_name,
                vessel=vessel,
                area_cm2=float(r.get("Area", 0.0)),
                mean_hu=float(r.get("Mean", 0.0)),
                max_hu=float(r.get("Max", 0.0)),
                min_hu=float(r.get("Min", 0.0)),
                total_hu=float(r.get("Total", 0.0)),
                n_points=n_pts,
                points_px=pts_px,
                points_mm=pts_mm,
                center_xyz=_parse_center(r.get("Center")),
            ))
            n_active += 1

        out_slices.append(SliceAnnotation(
            image_index=int(image_entry["ImageIndex"]),
            rois=tuple(out_rois),
        ))

    return ParseResult(
        pid=str(pid),
        slices=tuple(out_slices),
        dirty_vessel_names=tuple(sorted(dirty)),
        n_active_rois=n_active,
        n_dropped_zero_point=n_dropped,
    )
