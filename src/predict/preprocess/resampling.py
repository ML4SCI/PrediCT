"""Resample CT and mask to the target voxel grid.

The grid is set in ``configs/default.yaml`` (``resample.in_plane_mm`` and
``resample.slice_mm``). For COCA this is ``0.375 × 0.375 × 3.0 mm``, not
isotropic; the z dimension preserves native 3 mm slice thickness.

The in-plane target is the cohort native median (0.377 mm), so resampling
is minimum-interpolation. Nothing in this module hardcodes the value.

The CT uses B-spline interpolation; the mask uses nearest-neighbour.

Decisions referencing this module:
    D005: Target voxel grid (anisotropic by design)
"""
from __future__ import annotations

import SimpleITK as sitk


def resample_to_target(
    ct_sitk: sitk.Image,
    mask_sitk: sitk.Image | None,
    target_spacing: tuple[float, float, float],
    *,
    ct_default_value: float = -1000.0,
    mask_default_value: int = 0,
) -> tuple[sitk.Image, sitk.Image | None]:
    """Resample ``ct_sitk`` (and optional ``mask_sitk``) to ``target_spacing``.

    The output size is computed to preserve the physical extent:
    ``new_size = round(old_size * old_spacing / target_spacing)``.

    Parameters
    ----------
    ct_sitk : sitk.Image
        Source CT (any spacing).
    mask_sitk : sitk.Image | None
        Source mask aligned to ``ct_sitk``; spacing/origin/direction must
        match (callers use :func:`mask_to_sitk` to ensure this).
    target_spacing : tuple[float, float, float]
        Desired (x, y, z) spacing in mm.
    ct_default_value : float
        HU fill value outside the source FOV.
    mask_default_value : int
        Fill value (0 = background) outside the source FOV.
    """
    resampler = _target_resampler(ct_sitk, target_spacing)

    # CT: B-spline for smooth HU interpolation.
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(ct_default_value)
    ct_out = resampler.Execute(ct_sitk)

    if mask_sitk is None:
        return ct_out, None

    return ct_out, resample_mask_to_target(
        mask_sitk, ct_sitk, target_spacing, mask_default_value=mask_default_value,
    )


def _target_resampler(
    ref_sitk: sitk.Image,
    target_spacing: tuple[float, float, float],
) -> sitk.ResampleImageFilter:
    """Resampler onto the target grid of ``ref_sitk`` (extent preserved)."""
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(ref_sitk.GetSize(), ref_sitk.GetSpacing(), target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(ref_sitk.GetDirection())
    resampler.SetOutputOrigin(ref_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    return resampler


def resample_mask_to_target(
    mask_sitk: sitk.Image,
    ref_sitk: sitk.Image,
    target_spacing: tuple[float, float, float],
    *,
    mask_default_value: int = 0,
) -> sitk.Image:
    """Resample a mask aligned to ``ref_sitk`` onto the same target grid that
    :func:`resample_to_target` gives the CT. Nearest-neighbour keeps {0, 1}.

    Used alone by the D033 perturbation stage, which rebuilds the mask per
    draw but reuses the already-resampled CT.
    """
    resampler = _target_resampler(ref_sitk, target_spacing)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(mask_default_value)
    return resampler.Execute(mask_sitk)
