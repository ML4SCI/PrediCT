"""PyRadiomics wrapper for whole-mask and per-vessel feature extraction.

The `radiomics` package is imported lazily inside the functions that need it,
so this module can be imported in any environment without PyRadiomics
installed (useful for static analysis and partial CI).

Input contract:

  - ``ct_array``    : ``np.ndarray`` of shape ``(n_slices, H, W)`` in raw HU.
                       The pipeline produces this in stage 2 as ``int16``,
                       clipped to ``[clip_min, clip_max]``. PyRadiomics
                       accepts int16, int32, float32, float64.
  - ``mask_array``  : ``np.ndarray`` of same shape, ``uint8 {0, 1}``.
  - ``spacing``     : ``(sx, sy, sz)`` mm, the target voxel grid from
                       ``outputs/02_preprocessed/spacing.json``.

We wrap both arrays as ``SimpleITK.Image`` with matching spacing/origin/
direction (origin (0,0,0), identity direction: radiomics features are
translation- and rotation-invariant) and run the extractor.

Output:

  - ``dict[str, float]`` of feature_name → value. ``diagnostics_*`` keys are
    stripped. Empty masks return an empty dict (caller decides whether to
    record zeros or skip the patient).

Decisions referencing this module:
    D009: PyRadiomics extractor configuration (locked params.yaml).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk

from predict.config import REPO_ROOT

if TYPE_CHECKING:                                # pragma: no cover - type hints only
    from radiomics.featureextractor import RadiomicsFeatureExtractor


DEFAULT_PARAMS_YAML: Path = REPO_ROOT / "params.yaml"


# ───────────────────── extractor factory ─────────────────────


def create_extractor(params_yaml: Path = DEFAULT_PARAMS_YAML) -> "RadiomicsFeatureExtractor":
    """Instantiate a PyRadiomics extractor from a locked params.yaml.

    Sets PyRadiomics' logger to WARNING to suppress per-feature INFO chatter
    in worker processes.
    """
    import radiomics
    from radiomics import featureextractor

    radiomics.setVerbosity(logging.ERROR)
    # Also dampen the per-feature INFO logger that survives setVerbosity.
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return featureextractor.RadiomicsFeatureExtractor(str(params_yaml))


# ───────────────────── guardrail ─────────────────────


def validate_ct_for_radiomics(ct_array: np.ndarray, pid: str = "") -> None:
    """Fail fast if ``ct_array`` is not raw HU suitable for PyRadiomics.

    Catches:
      * normalised inputs (max ≤ 1.5): common bug if the windowed display
        volume is fed in by accident.
      * min HU well above air (≥ −100): implies clipping that has hidden
        the air baseline; not strictly fatal but a strong smell.
      * max HU below 130: no calcium possible, downstream texture is
        guaranteed to be meaningless.
      * unexpected dtype.
    """
    valid_dtypes = (np.int16, np.int32, np.int64,
                    np.float32, np.float64)
    if ct_array.dtype.type not in valid_dtypes:
        raise ValueError(
            f"[GUARDRAIL] {pid}: CT dtype {ct_array.dtype} unexpected; "
            "PyRadiomics expects int/float."
        )

    cmin, cmax = float(ct_array.min()), float(ct_array.max())
    if cmax <= 1.5:
        raise ValueError(
            f"[GUARDRAIL] {pid}: CT max={cmax:.4f} suggests normalised input, "
            "not raw HU."
        )
    if cmin >= -100:
        raise ValueError(
            f"[GUARDRAIL] {pid}: CT min={cmin:.1f} too high for raw HU "
            "(expected near -1000 / clip_min)."
        )
    if cmax < 130:
        raise ValueError(
            f"[GUARDRAIL] {pid}: CT max={cmax:.1f} below CAC threshold (130 HU); "
            "calcium would be invisible."
        )


# ───────────────────── ROI size pre-check (D010) ─────────────────────


def roi_too_small(mask_array: np.ndarray, extractor: "RadiomicsFeatureExtractor") -> str | None:
    """Return why PyRadiomics' own ROI checks would reject ``mask_array``,
    or ``None`` if it is large enough.

    Mirrors PyRadiomics' ``imageoperations.checkMask`` boundaries exactly,
    with the extractor's locked settings:

    * size: rejected when the voxel count is **at or below**
      ``minimumROISize`` (PyRadiomics tests ``roiSize <= minimumROISize``,
      so with 14 a 14-voxel mask is rejected and 15 is accepted);
    * dimensions: rejected for a single voxel (0 axes with extent > 1,
      always), or when fewer axes than ``minimumROIDimensions`` (default 2)
      have extent > 1.

    Deciding this before extraction means the caller never has to guess
    from an exception message: a legitimate small-ROI skip is this
    function's answer, and anything the extractor raises is a real failure.
    PyRadiomics rejects a mask at or below the minimum size (``<=``, not
    ``<``: a 14-voxel mask raises inside PyRadiomics);
    ``tests/features/test_radiomics.py`` pins both sides of the boundary
    against PyRadiomics itself.
    """
    settings = getattr(extractor, "settings", {}) or {}
    n = int(np.count_nonzero(mask_array))
    min_size = settings.get("minimumROISize")
    if min_size is not None and n <= min_size:
        return f"roi_too_small: {n} voxels <= minimumROISize {min_size}"
    min_dims = settings.get("minimumROIDimensions", 2)
    nz = np.nonzero(mask_array)
    dims = sum(1 for axis in nz if axis.size and axis.max() > axis.min())
    if dims == 0:
        return "roi_too_small: single voxel (0 dimensions)"
    if dims < min_dims:
        return f"roi_too_small: {dims} dimension(s) < minimumROIDimensions {min_dims}"
    return None


# ───────────────────── core extraction ─────────────────────


def _arrays_to_sitk(
    ct_array: np.ndarray,
    mask_array: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[sitk.Image, sitk.Image]:
    """Wrap CT + mask as SITK images with matching geometry.

    ``spacing`` is the (x, y, z) physical voxel size in mm (the canonical
    SimpleITK order, matching the layout in ``predict.io.spacing``).
    """
    if ct_array.shape != mask_array.shape:
        raise ValueError(
            f"Shape mismatch: ct {ct_array.shape} vs mask {mask_array.shape}"
        )
    ct_sitk = sitk.GetImageFromArray(ct_array.astype(np.int16, copy=False))
    mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8, copy=False))
    ct_sitk.SetSpacing(tuple(float(s) for s in spacing))
    mask_sitk.SetSpacing(tuple(float(s) for s in spacing))
    ct_sitk.SetOrigin((0.0, 0.0, 0.0))
    mask_sitk.SetOrigin((0.0, 0.0, 0.0))
    return ct_sitk, mask_sitk


def extract_pyradiomics(
    ct_array: np.ndarray,
    mask_array: np.ndarray,
    spacing: tuple[float, float, float],
    extractor: "RadiomicsFeatureExtractor",
    *,
    label: int = 1,
    pid: str = "",
) -> dict[str, float]:
    """Run PyRadiomics on (ct, mask) and return a flat ``{name: value}`` dict.

    Returns an empty dict if the mask is empty. Raises any PyRadiomics
    exception to the caller: the orchestration script catches and logs.

    The ``diagnostics_*`` keys produced by PyRadiomics are stripped.
    """
    if mask_array.sum() == 0:
        return {}

    validate_ct_for_radiomics(ct_array, pid=pid)
    ct_sitk, mask_sitk = _arrays_to_sitk(ct_array, mask_array, spacing)
    result = extractor.execute(ct_sitk, mask_sitk, label=label)

    features: dict[str, float] = {}
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            features[k] = float(v)
        except (TypeError, ValueError):
            # Some PyRadiomics outputs are sitk types or arrays; convert
            # what we can, drop the rest.
            try:
                features[k] = float(np.asarray(v).squeeze())
            except Exception:
                continue
    return features


