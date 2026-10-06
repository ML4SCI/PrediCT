"""Spacing metadata I/O for preprocessed CT volumes.

A single JSON file alongside the preprocessed ``.npy`` volumes records the
target voxel grid so that downstream radiomics extraction reads it instead
of guessing from disk.

Decisions referencing this module:
    D005: Target voxel grid
"""
from __future__ import annotations

import json
from pathlib import Path


def parse_spacing(value: str) -> tuple[float, float, float]:
    """Parse a comma-separated ``x,y,z`` spacing string in millimetres."""
    try:
        parts = tuple(float(p.strip()) for p in value.split(","))
    except ValueError as exc:
        raise ValueError(
            f"Invalid spacing {value!r}; expected three comma-separated numbers."
        ) from exc

    if len(parts) != 3:
        raise ValueError(f"Invalid spacing {value!r}; expected exactly three values.")
    if any(p <= 0 for p in parts):
        raise ValueError(f"Invalid spacing {value!r}; all values must be positive.")
    return parts


def save_spacing_metadata(path: Path, target_spacing: tuple[float, float, float]) -> None:
    """Persist preprocessing spacing metadata as JSON."""
    payload = {
        "target_spacing": list(target_spacing),
        "spacing_unit": "mm",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_spacing_metadata(path: Path) -> tuple[float, float, float]:
    """Load preprocessing spacing metadata from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "target_spacing" not in payload:
        raise ValueError(f"{path} does not contain target_spacing.")
    spacing = payload["target_spacing"]
    if not isinstance(spacing, list) or len(spacing) != 3:
        raise ValueError(f"{path} target_spacing must be a 3-element JSON list.")
    return parse_spacing(",".join(str(p) for p in spacing))
