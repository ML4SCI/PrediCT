"""Patient-level spatial features derived from grouped lesions.

Inputs: ``dict[vessel -> list[Lesion]]`` from :mod:`predict.features.lesion_ccl`,
plus the patient's ``slice_positions`` (for ``dist_from_top_*``) and slice
thickness (not currently needed but kept in the signature for forward
compatibility with cohorts where lesion volume should be re-derived).

Output (26 keys):

  lesion_count_{vessel}, lesion_count_total          (5)
  inter_lesion_dist_mean_{vessel}                    (4)
  inter_lesion_dist_max_{vessel}                     (4)
  first_to_last_dist_{vessel}                        (4)
  diffusivity_{vessel}                               (4)
  n_calcified_arteries                               (1)
  gini_lesion_volume                                 (1)
  dist_from_top_max, dist_from_top_mean              (2)
  center_of_mass_z                                   (1)

Edge cases:

  • Distance keys are 0.0 when the vessel has fewer than 2 lesions (D017).
  • Diffusivity follows D016: N=0 → 0; N=1 → 1; N≥2 with d≈0 → 1; else N/d.
  • Gini is 0.0 for < 2 lesions or all-equal volumes.
  • ``dist_from_top_*`` use the most-superior physical Z (= ``max(slice_positions)``)
    minus the lesion centroid Z, in mm.

Decisions referencing this module:
    D007: Lesion grouping (provides the lesions consumed here).
    D016: Diffusivity edge-case table.
    D017: Zero sentinel for empty-artery / insufficient-data features.
"""
from __future__ import annotations

from typing import Sequence

from predict.config import VESSEL_NAMES
from predict.features.lesion_ccl import Lesion


DIFFUSIVITY_ZERO_DISTANCE_EPSILON_MM: float = 1e-6


# ───────────────────── pure math helpers ─────────────────────


def gini_coefficient(values: Sequence[float]) -> float:
    """Standard Gini over non-negative values.

    Returns 0.0 for empty / single-value inputs and for all-zero inputs.
    """
    if len(values) < 2:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2.0 * cum) / (n * total) - (n + 1) / n


def diffusivity(n_lesions: int, d_first_last_mm: float) -> float:
    """Hoori 2024 diffusivity with D016 edge cases."""
    if n_lesions == 0:
        return 0.0
    if n_lesions == 1:
        return 1.0
    if d_first_last_mm < DIFFUSIVITY_ZERO_DISTANCE_EPSILON_MM:
        return 1.0
    return n_lesions / d_first_last_mm


def _euclid3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _sort_lesions_by_z(lesions: list[Lesion]) -> list[Lesion]:
    """Sort by centroid z asc, then x asc, then y asc (deterministic)."""
    return sorted(lesions, key=lambda l: (l.centroid_mm[2], l.centroid_mm[0], l.centroid_mm[1]))


# ───────────────────── feature computations ─────────────────────


def _per_vessel_counts_and_distances(
    lesions_per_vessel: dict[str, list[Lesion]],
) -> dict[str, float]:
    """Per-vessel lesion counts, inter-lesion + first-to-last distances,
    and diffusivity, into a flat feature dict."""
    out: dict[str, float] = {}
    total_count = 0

    for vessel in VESSEL_NAMES:
        suf = vessel.lower()
        lesions = _sort_lesions_by_z(lesions_per_vessel.get(vessel, []))
        n = len(lesions)
        total_count += n

        out[f"lesion_count_{suf}"] = float(n)

        if n < 2:
            out[f"inter_lesion_dist_mean_{suf}"] = 0.0
            out[f"inter_lesion_dist_max_{suf}"] = 0.0
            out[f"first_to_last_dist_{suf}"] = 0.0
            out[f"diffusivity_{suf}"] = diffusivity(n, 0.0)
            continue

        consecutive = [
            _euclid3(lesions[i].centroid_mm, lesions[i + 1].centroid_mm)
            for i in range(n - 1)
        ]
        first_to_last = _euclid3(lesions[0].centroid_mm, lesions[-1].centroid_mm)

        out[f"inter_lesion_dist_mean_{suf}"] = float(sum(consecutive) / len(consecutive))
        out[f"inter_lesion_dist_max_{suf}"] = float(max(consecutive))
        out[f"first_to_last_dist_{suf}"] = float(first_to_last)
        out[f"diffusivity_{suf}"] = diffusivity(n, first_to_last)

    out["lesion_count_total"] = float(total_count)
    return out


def _global_spatial_summaries(
    lesions_per_vessel: dict[str, list[Lesion]],
    slice_positions: Sequence[float],
) -> dict[str, float]:
    """n_calcified_arteries, gini_lesion_volume, dist_from_top_*, center_of_mass_z."""
    all_lesions: list[Lesion] = [
        l for vessel in VESSEL_NAMES for l in lesions_per_vessel.get(vessel, [])
    ]
    n_calcified = sum(1 for v in VESSEL_NAMES if lesions_per_vessel.get(v))

    if not all_lesions:
        return {
            "n_calcified_arteries": 0.0,
            "gini_lesion_volume": 0.0,
            "dist_from_top_max": 0.0,
            "dist_from_top_mean": 0.0,
            "center_of_mass_z": 0.0,
        }

    z_top = max(slice_positions) if slice_positions else 0.0
    dist_from_top = [z_top - l.centroid_mm[2] for l in all_lesions]
    # Clamp tiny negatives (which can only come from float roundoff at the top slice).
    dist_from_top = [max(0.0, d) for d in dist_from_top]

    total_area = sum(l.total_area_mm2 for l in all_lesions)
    if total_area > 0:
        weighted_mean_z = (
            sum(l.total_area_mm2 * l.centroid_mm[2] for l in all_lesions) / total_area
        )
    else:
        weighted_mean_z = sum(l.centroid_mm[2] for l in all_lesions) / len(all_lesions)

    # Patient-relative: distance from the most-superior slice to the
    # weighted centre of mass. Same convention as dist_from_top_*: invariant
    # to the scanner's absolute IPP[2] origin. Clamp tiny float-roundoff
    # negatives to 0.
    center_of_mass_z = max(0.0, z_top - weighted_mean_z)

    return {
        "n_calcified_arteries": float(n_calcified),
        "gini_lesion_volume": gini_coefficient([l.volume_mm3 for l in all_lesions]),
        "dist_from_top_max": float(max(dist_from_top)),
        "dist_from_top_mean": float(sum(dist_from_top) / len(dist_from_top)),
        "center_of_mass_z": float(center_of_mass_z),
    }


def compute_spatial_features(
    lesions_per_vessel: dict[str, list[Lesion]],
    *,
    slice_positions: Sequence[float],
    slice_thickness_mm: float = 3.0,        # kept for forward compatibility
) -> dict[str, float]:
    """Return all patient-level spatial scalars (26 keys)."""
    out = _per_vessel_counts_and_distances(lesions_per_vessel)
    out.update(_global_spatial_summaries(lesions_per_vessel, slice_positions))
    return out
