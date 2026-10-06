"""HU clipping and metal-artifact detection.

No display window; radiomics consumes raw HU. Clipping is applied at a wide
range (default ``[-200, 3000]``) so the dense-calcium tail is preserved.
Patients with mask voxels above the metal-artifact threshold are flagged for
downstream review (some texture features become unreliable in the presence
of pacemaker/stent leads).

Decisions referencing this module:
    D003: No display HU window in pipeline outputs
"""
from __future__ import annotations

import numpy as np


def clip_hu(ct_array: np.ndarray, clip_min: int, clip_max: int) -> np.ndarray:
    """Clip HU values into ``[clip_min, clip_max]``.

    Returns a new array; the input is not modified.
    """
    if clip_min >= clip_max:
        raise ValueError(f"clip_min ({clip_min}) must be < clip_max ({clip_max})")
    return np.clip(ct_array, clip_min, clip_max)


def flag_metal_artifact(
    ct_array: np.ndarray,
    mask_array: np.ndarray,
    threshold: int,
) -> bool:
    """Return True if any voxel inside the calcium mask exceeds ``threshold`` HU.

    Used to flag patients whose calcium mask overlaps a metal-density region
    (pacemaker leads, surgical clips). The orchestration script logs the
    flag and may exclude the patient from texture extraction depending on
    configuration.
    """
    if mask_array.shape != ct_array.shape:
        raise ValueError(
            f"Shape mismatch: ct {ct_array.shape} vs mask {mask_array.shape}"
        )
    if not mask_array.any():
        return False
    return bool(ct_array[mask_array.astype(bool)].max() > threshold)
