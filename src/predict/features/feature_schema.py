"""Canonical feature-name registry for the features stage.

This module is the single source of truth for which scalar features the
features-stage CSV exposes. Every other feature module either:

- returns a subset whose keys are a subset of :func:`zero_features`'s keys, or
- is asserted against this schema in its own unit tests.

The orchestration script writes ``outputs/03_features/features.csv`` with
columns equal to ``zero_features()`` keys + identification + metadata
(``pid``, ``kernel``, ``mask_voxels``, ``low_burden_flag``,
``roundtrip_quality``, ``category``) + PyRadiomics features prefixed
``original_<family>_<name>``.

The function returns a fresh dict on each call: callers may mutate it.

Decisions referencing this module:
    D007: Lesion grouping rule
    D008: Per-artery masks
    D010: low_burden_flag (orchestration column)
    D012: excluded_roi_ids (does not change the schema, only the values)
"""
from __future__ import annotations

from predict.config import VESSEL_NAMES


# Canonical lowercase suffixes used in feature names. Matches VESSEL_NAMES
# elementwise: the upper-case name is the canonical token; the suffix is the
# lowercase form used in column names.
VESSEL_SUFFIXES: tuple[str, ...] = tuple(v.lower() for v in VESSEL_NAMES)

# Density tiers: Agatston factor groups per D009 / D024.
DENSITY_TIERS: tuple[str, ...] = ("d1", "d2", "d3", "d4")

# Per-vessel feature stems (suffix is appended at build time).
PER_VESSEL_STEMS: tuple[str, ...] = (
    "lesion_count",
    "max_hu",
    "mean_hu",
    "volume_mm3",
    "mass",
    "agatston",
    "inter_lesion_dist_mean",
    "inter_lesion_dist_max",
    "first_to_last_dist",
    "diffusivity",
)

# Global patient-level features.
GLOBAL_FEATURES: tuple[str, ...] = (
    "lesion_count_total",
    "n_calcified_arteries",
    "gini_lesion_volume",
    "dist_from_top_max",
    "dist_from_top_mean",
    "dense_calcium_count",
    "agatston_total",
    "volume_total_mm3",
    "mass_total",
    "mean_hu_weighted_global",
    "max_hu_global",
    "center_of_mass_z",
)


def _per_vessel_keys() -> list[str]:
    """Build the per-vessel feature key list deterministically.

    Layout: for each vessel suffix in ``VESSEL_SUFFIXES`` order, emit all
    ``PER_VESSEL_STEMS`` then the 4 density-tier keys.
    """
    keys: list[str] = []
    for suf in VESSEL_SUFFIXES:
        for stem in PER_VESSEL_STEMS:
            if stem == "volume_mm3":
                keys.append(f"volume_{suf}_mm3")
            else:
                keys.append(f"{stem}_{suf}")
        for tier in DENSITY_TIERS:
            keys.append(f"n_rois_{tier}_{suf}")
    return keys


def feature_names() -> tuple[str, ...]:
    """Return the canonical feature name tuple, in writing order."""
    return tuple(_per_vessel_keys()) + GLOBAL_FEATURES


def zero_features() -> dict[str, float]:
    """Return a fresh ``{name: 0.0}`` dict containing every canonical key.

    Use this as the starting point in any feature-extraction function so the
    output is guaranteed schema-complete. Patients with no lesions in a given
    vessel get the zero sentinel (D017) for that vessel's keys.
    """
    return {name: 0.0 for name in feature_names()}


def n_features() -> int:
    """Number of scalar features in the canonical schema."""
    return len(feature_names())
