"""Canonical CT perturbations for stage-4 ICC stability analysis (D014).

Each patient's preprocessed CT volume is transformed in 14 deterministic ways.
The mask is held fixed (D014), so the perturbations
stress mis-registration robustness rather than extractor equivariance.

Public API:

* :func:`enumerate_perturbations` returns the locked 14 perturbation specs.
* :func:`apply_perturbation` dispatches one spec to the appropriate transform.
* :func:`rotate`, :func:`translate`, :func:`add_gaussian_noise` are the
  individual transforms exposed for direct use and testing.
* :func:`noise_seed` derives the deterministic per-(patient, sigma) noise seed.

Conventions:

* Spatial transforms (rotate, translate) use SimpleITK with linear
  interpolation and a constant background fill of ``stability.background_fill_hu``
  (default -1024, the standard outside-FOV sentinel).
* Noise is additive Gaussian, clipped to the same HU range
  (``[hu.clip_min, hu.clip_max]``) as the source preprocessed array. The
  per-patient seed is ``int(pid) * stability.noise_seed_multiplier +
  int(sigma)`` so the same patient produces the same noisy array on rerun.
* All transforms preserve volume shape, spacing, origin, and direction;
  only voxel values change. The mask is never returned (it is the caller's
  responsibility to pass the unchanged mask to the feature extractor).

Decisions referencing this module:
    D013 - ICC formulation and threshold (consumer of the perturbation outputs).
    D014 - Perturbation set (this module is the implementation).
    D016 - Geometric bypass (HU-touching features go through this module;
            pure-geometry features bypass).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import SimpleITK as sitk

from predict.config import Config
from predict.io.spacing import load_spacing_metadata


# ───────────────────────── perturbation specification ─────────────────────────


PerturbationKind = Literal["rotate", "translate", "noise"]


@dataclass(frozen=True)
class PerturbationSpec:
    """Frozen description of one perturbation.

    `name` is the stable string key used in every output file and in the
    reliability matrix column index. The format is fixed and tests assert
    on it; do not rename without bumping a decision doc.
    """
    name: str
    kind: PerturbationKind
    # rotation
    rotation_deg: float = 0.0
    # translation
    tx_mm: float = 0.0
    ty_mm: float = 0.0
    # noise
    sigma_hu: float = 0.0


def _fmt_signed(x: float) -> str:
    """Format a signed float for use in perturbation names (e.g. +5, -10)."""
    return f"{x:+g}"


def enumerate_perturbations(cfg: Config) -> tuple[PerturbationSpec, ...]:
    """Return the locked 14-perturbation set in deterministic order.

    Order: all rotations (ascending magnitude, +sign before -sign within each
    magnitude), all translations (x before y within each magnitude, +sign
    before -sign within each axis), then noise (ascending sigma). This order
    is stable across reruns and is asserted in tests.
    """
    specs: list[PerturbationSpec] = []

    for deg in cfg.stability.rotation_degrees:
        for sign in (+1.0, -1.0):
            angle = sign * float(deg)
            specs.append(PerturbationSpec(
                name=f"rotate_{_fmt_signed(angle)}",
                kind="rotate",
                rotation_deg=angle,
            ))

    for mm in cfg.stability.translation_mm:
        for axis in ("x", "y"):
            for sign in (+1.0, -1.0):
                offset = sign * float(mm)
                tx = offset if axis == "x" else 0.0
                ty = offset if axis == "y" else 0.0
                specs.append(PerturbationSpec(
                    name=f"translate_{_fmt_signed(offset)}_{axis}",
                    kind="translate",
                    tx_mm=tx,
                    ty_mm=ty,
                ))

    for sigma in cfg.stability.noise_sigma_hu:
        specs.append(PerturbationSpec(
            name=f"noise_{float(sigma):g}",
            kind="noise",
            sigma_hu=float(sigma),
        ))

    expected = (
        2 * len(cfg.stability.rotation_degrees)
        + 4 * len(cfg.stability.translation_mm)
        + len(cfg.stability.noise_sigma_hu)
    )
    if len(specs) != expected:
        raise RuntimeError(
            f"enumerate_perturbations: built {len(specs)} specs, expected {expected}"
        )
    if expected != 14:
        raise RuntimeError(
            f"D014 lock requires exactly 14 perturbations, got {expected}. "
            f"Check configs/default.yaml stability section."
        )
    return tuple(specs)


# ───────────────────────── helpers ─────────────────────────


def _volume_center_physical(image: sitk.Image) -> tuple[float, float, float]:
    """Physical-space (x, y, z) coordinates of the volume centre voxel."""
    center_index = [(size - 1) / 2.0 for size in image.GetSize()]
    return image.TransformContinuousIndexToPhysicalPoint(center_index)


def noise_seed(pid: str | int, sigma_hu: float, multiplier: int) -> int:
    """Deterministic per-(patient, sigma) seed.

    `pid` may be a numeric string (e.g. "306") or already an int; we coerce.
    If the pid is non-numeric (shouldn't happen in COCA, but kept robust), we
    fall back to a hash so reruns are still reproducible.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        pid_int = abs(hash(str(pid))) % (2 ** 31)
    return pid_int * int(multiplier) + int(round(sigma_hu))


# ───────────────────────── transforms ─────────────────────────


def rotate(
    ct_img: sitk.Image,
    degrees: float,
    *,
    background_hu: float = -1024.0,
) -> sitk.Image:
    """Rotate the CT volume about its centre on the z (axial) axis.

    Returns a new image with identical size, spacing, origin, and direction.
    The mask is NOT transformed (D014). Linear interpolation, constant
    background fill = ``background_hu`` for voxels rotated in from outside the
    field of view.
    """
    transform = sitk.Euler3DTransform()
    transform.SetCenter(_volume_center_physical(ct_img))
    transform.SetRotation(0.0, 0.0, float(np.deg2rad(degrees)))
    # SimpleITK Resample maps output coords back to input via the supplied
    # transform; to rotate image content by +theta we resample with the inverse.
    inv = transform.GetInverse()
    return sitk.Resample(
        ct_img, ct_img, inv, sitk.sitkLinear,
        float(background_hu), ct_img.GetPixelID(),
    )


def translate(
    ct_img: sitk.Image,
    tx_mm: float,
    ty_mm: float,
    *,
    background_hu: float = -1024.0,
) -> sitk.Image:
    """Translate the CT volume in physical x/y by the given millimetres.

    z is never translated (D014: slice spacing 3 mm makes sub-slice shifts
    ambiguous). Returns a new image with identical geometry metadata.
    """
    transform = sitk.TranslationTransform(3, (float(tx_mm), float(ty_mm), 0.0))
    inv = transform.GetInverse()
    return sitk.Resample(
        ct_img, ct_img, inv, sitk.sitkLinear,
        float(background_hu), ct_img.GetPixelID(),
    )


def add_gaussian_noise(
    ct_img: sitk.Image,
    sigma_hu: float,
    *,
    seed: int,
    clip_min: float = -200.0,
    clip_max: float = 3000.0,
) -> sitk.Image:
    """Add deterministic per-voxel Gaussian noise (mean 0, std sigma_hu).

    Clipped to ``[clip_min, clip_max]`` matching the source preprocessed range.
    Spacing / origin / direction are preserved.
    """
    rng = np.random.default_rng(int(seed))
    arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    noise = rng.normal(0.0, float(sigma_hu), size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, float(clip_min), float(clip_max))
    out = sitk.GetImageFromArray(noisy)
    out.CopyInformation(ct_img)
    return out


# ───────────────────────── dispatch ─────────────────────────


def apply_perturbation(
    ct_img: sitk.Image,
    spec: PerturbationSpec,
    *,
    pid: str | int,
    cfg: Config,
) -> sitk.Image:
    """Dispatch one perturbation spec, returning the perturbed CT.

    The mask is NOT modified. Callers pass the original mask straight through
    to the feature extractor alongside the perturbed CT.
    """
    background = cfg.stability.background_fill_hu

    if spec.kind == "rotate":
        return rotate(ct_img, spec.rotation_deg, background_hu=background)
    if spec.kind == "translate":
        return translate(
            ct_img, spec.tx_mm, spec.ty_mm, background_hu=background,
        )
    if spec.kind == "noise":
        seed = noise_seed(pid, spec.sigma_hu, cfg.stability.noise_seed_multiplier)
        return add_gaussian_noise(
            ct_img,
            spec.sigma_hu,
            seed=seed,
            clip_min=cfg.hu.clip_min,
            clip_max=cfg.hu.clip_max,
        )
    raise ValueError(f"Unknown perturbation kind: {spec.kind!r}")


# ───────────────────────── I/O convenience ─────────────────────────


def load_preprocessed_ct_and_mask(
    pid: str,
    cfg: Config,
) -> tuple[sitk.Image, sitk.Image]:
    """Load the stage-2 preprocessed CT + mask for a patient as SimpleITK images.

    Spacing is read from ``outputs/02_preprocessed/spacing.json``.
    """
    preproc = cfg.paths.outputs / "02_preprocessed"
    ct_np = np.load(preproc / f"{pid}_ct.npy")
    mask_np = np.load(preproc / f"{pid}_mask.npy")

    # spacing.json convention (predict.io.spacing): SimpleITK (x, y, z) in mm.
    spacing = load_spacing_metadata(preproc / "spacing.json")

    ct_img = sitk.GetImageFromArray(ct_np.astype(np.float32))
    ct_img.SetSpacing(spacing)
    mask_img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    mask_img.SetSpacing(spacing)
    return ct_img, mask_img


__all__ = [
    "PerturbationSpec",
    "enumerate_perturbations",
    "noise_seed",
    "rotate",
    "translate",
    "add_gaussian_noise",
    "apply_perturbation",
    "load_preprocessed_ct_and_mask",
]
