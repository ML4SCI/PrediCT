"""D033 realistic composite perturbations (primary stability gate).

Every rule here implements a line of ``decisions/D033-realistic-perturbation-set.md``.
If the two ever disagree, the decision document wins and this module is the bug.

Structure
---------
* :class:`Calibration` holds the image-derived constants from
  ``scripts/calibrate_realistic.py`` (C1 sigma0 per patient, C1b noise
  correlation per kernel, C2 sigma_delta, C3 native-spacing distribution).
* :func:`sample_draw` turns (pid, arm, draw index) into a frozen :class:`Draw`
  of factor magnitudes. Every factor is always drawn, in the fixed order
  M1, P1..P7, then the secondary extras, so a draw is reproducible in
  isolation and switching a factor off never shifts the random stream.
* :func:`jitter_parse_result` is M1: smooth normal displacement of every XML
  polygon at native resolution. Canonical features and the mask are then
  rebuilt from the jittered annotation by the production code.
* :func:`apply_draw_to_volumes` applies the rest in D033 order: P1-P3 as one
  composed resampling per image, P4, P5, S2, P6, P7, clip, then the secondary
  mask rules S3 / S4.

Zero magnitudes are exact identities (no resampling happens), which is what
lets stage 4b check parity against the stage-2 mask and stage-3 features.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from predict.io.xml_parser import ParseResult

PRIMARY_FACTORS: tuple[str, ...] = ("M1", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
SECONDARY_FACTORS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")

N_PRIMARY_DRAWS = 20
N_SECONDARY_DRAWS = 10
N_ATTRIBUTION_DRAWS = 10

# D033 magnitudes. Changing any of these changes a locked decision.
M1_RMS_MAX_PX = 0.5
M1_CAP_PX = 1.0
M1_AREA_FLOOR = 0.5
M1_FOURIER_ORDER = 3
P1_MAX_DEG = 5.0
P2_MAX_VOX = 0.5
P3_MAX_VOX = 1.0
P5_MIN_SIGMA_DELTA_VOX = 0.1
P6_DOSE_RANGE = (0.5, 1.0)
P7_MAX_HU = 5.0
S1_MAX_MM = 1.5
S2_MAX_MM = 1.5
S5_DOSE_RANGE = (0.25, 0.5)
S6_MAX_SPACING_MM = 0.90
CALCIUM_HU = 130.0
AGATSTON_MIN_AREA_MM2 = 1.0

SOFT_BAND = (-100.0, 200.0)
_RNG_TAG = 33


# ───────────────────────── calibration ─────────────────────────


@dataclass(frozen=True)
class Calibration:
    sigma0: dict[str, float]          # pid -> image noise sd (HU), C1
    sigma_n: dict[str, float]         # kernel -> noise correlation sd (voxels), C1b
    sigma_n_default: float            # pooled median, for kernels outside the two main ones
    sigma_delta: float                # voxels, C2
    native_spacings: tuple[float, ...]  # C3, eligible cohort

    @classmethod
    def from_json(cls, path: Path) -> "Calibration":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            sigma0={str(k): float(v) for k, v in d["sigma0"].items()},
            sigma_n={str(k): float(v) for k, v in d["sigma_n"].items()},
            sigma_n_default=float(d["sigma_n_default"]),
            sigma_delta=float(d["sigma_delta"]),
            native_spacings=tuple(float(s) for s in d["native_spacings"]),
        )

    def noise_corr(self, kernel: str) -> float:
        return self.sigma_n.get(kernel, self.sigma_n_default)


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def immerkaer_sigma(vol: np.ndarray, band=SOFT_BAND, min_px: int = 2000) -> float:
    """C1: median over slices of the Immerkaer (1996) noise sd inside a HU band.

    A pixel counts only if its whole 3x3 neighbourhood is inside the band, so
    tissue edges do not inflate the estimate.
    """
    from scipy.ndimage import binary_erosion, convolve
    lo, hi = band
    sig = []
    for sl in np.asarray(vol, dtype=np.float64):
        valid = binary_erosion((sl >= lo) & (sl <= hi), structure=np.ones((3, 3), bool))
        if valid.sum() < min_px:
            continue
        r = convolve(sl, _IMMERKAER, mode="nearest")
        sig.append(math.sqrt(math.pi / 2.0) * np.abs(r[valid]).mean() / 6.0)
    return float(np.median(sig)) if sig else float("nan")


def noise_lag1_correlation(vol: np.ndarray, band=SOFT_BAND, min_pairs: int = 2000) -> float:
    """C1b: lag-1 (x) autocorrelation of the high-pass residual in the band.

    Residual = CT minus its 5x5 in-plane median. Only pixel pairs whose full
    5x5 neighbourhoods lie in the band are used.
    """
    from scipy.ndimage import binary_erosion, median_filter
    lo, hi = band
    a_all, b_all = [], []
    for sl in np.asarray(vol, dtype=np.float64):
        valid = binary_erosion((sl >= lo) & (sl <= hi), structure=np.ones((5, 5), bool))
        if valid.sum() < 100:
            continue
        res = sl - median_filter(sl, size=5, mode="nearest")
        pair = valid[:, :-1] & valid[:, 1:]
        a_all.append(res[:, :-1][pair])
        b_all.append(res[:, 1:][pair])
    if not a_all:
        return float("nan")
    a, b = np.concatenate(a_all), np.concatenate(b_all)
    if a.size < min_pairs:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def lag1_of_filtered_white_noise(sigma_vox: float) -> float:
    """Lag-1 autocorrelation of white noise after scipy's discrete Gaussian
    filter of sd ``sigma_vox`` (truncate=4, the scipy default used by
    :func:`correlated_noise`). The continuous limit is exp(-1 / (4 sigma^2));
    the discrete kernel differs at sub-voxel sigma, so we use it exactly."""
    if sigma_vox <= 0.0:
        return 0.0
    r = int(4.0 * sigma_vox + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma_vox) ** 2)
    w /= w.sum()
    return float(np.dot(w[:-1], w[1:]) / np.dot(w, w))


def sigma_from_lag1(rho1: float) -> float:
    """C1b: the filter sd (voxels) whose filtered white noise has lag-1
    autocorrelation ``rho1`` (bisection on the exact discrete relation)."""
    if not np.isfinite(rho1) or rho1 <= 0.0:
        return 0.0
    rho1 = min(rho1, 0.995)
    lo, hi = 1e-3, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if lag1_of_filtered_white_noise(mid) < rho1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ───────────────────────── draws ─────────────────────────


@dataclass(frozen=True)
class Draw:
    arm: str                 # "baseline", "primary", "S1".."S6", "attr_<factor>"
    index: int
    m1_rms_px: float = 0.0
    m1_seed: int = 0
    theta_deg: float = 0.0
    phase_dx_vox: float = 0.0
    phase_dy_vox: float = 0.0
    mis_dir_deg: float = 0.0
    mis_len_vox: float = 0.0
    s_star_mm: float = 0.0   # 0 = no resolution change
    sigma_b_vox: float = 0.0
    dose_f: float = 1.0      # 1 = no added noise
    noise_seed: int = 0
    offset_hu: float = 0.0
    dz_mm: float = 0.0       # S1
    motion_len_mm: float = 0.0  # S2
    motion_dir_deg: float = 0.0
    m2: bool = False         # S3
    m3: bool = False         # S4

    def as_record(self) -> dict:
        return dict(self.__dict__)


def baseline_draw() -> Draw:
    return Draw(arm="baseline", index=-1)


def _arm_code(arm: str) -> int:
    order = ("primary",) + SECONDARY_FACTORS + tuple(f"attr_{f}" for f in PRIMARY_FACTORS)
    return order.index(arm)


def sample_draw(
    pid: str,
    arm: str,
    index: int,
    calib: Calibration,
    native_spacing_mm: float,
) -> Draw:
    """Magnitudes for one draw. Primary seeds are exactly D033's
    ``default_rng([int(pid), draw_index, 33])``; other arms append an arm code."""
    seed = [int(pid), int(index), _RNG_TAG]
    if arm != "primary":
        seed.append(_arm_code(arm))
    rng = np.random.default_rng(seed)

    # Fixed order: M1, P1..P7, then secondary extras. Always all drawn.
    m1_rms = rng.uniform(0.0, M1_RMS_MAX_PX)
    m1_seed = int(rng.integers(0, 2**31 - 1))
    theta = rng.uniform(-P1_MAX_DEG, P1_MAX_DEG)
    pdx = rng.uniform(-P2_MAX_VOX, P2_MAX_VOX)
    pdy = rng.uniform(-P2_MAX_VOX, P2_MAX_VOX)
    mis_dir = rng.uniform(0.0, 360.0)
    mis_len = rng.uniform(0.0, P3_MAX_VOX)
    s_pick = float(rng.choice(np.asarray(calib.native_spacings)))
    sb = rng.uniform(0.0, calib.sigma_delta) if calib.sigma_delta >= P5_MIN_SIGMA_DELTA_VOX else 0.0
    dose = rng.uniform(*P6_DOSE_RANGE)
    noise_seed = int(rng.integers(0, 2**31 - 1))
    offset = rng.uniform(-P7_MAX_HU, P7_MAX_HU)
    dz = rng.uniform(-S1_MAX_MM, S1_MAX_MM)
    mot_len = rng.uniform(0.0, S2_MAX_MM)
    mot_dir = rng.uniform(0.0, 180.0)
    dose_low = rng.uniform(*S5_DOSE_RANGE)
    s6_u = rng.uniform(0.0, 1.0)

    # P4: an image is never made sharper than it was acquired.
    s_star = s_pick if s_pick > native_spacing_mm + 1e-9 else 0.0

    if arm.startswith("attr_"):
        active = {arm[len("attr_"):]}
    elif arm in ("primary",) + SECONDARY_FACTORS:
        active = set(PRIMARY_FACTORS)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    d = dict(
        arm=arm, index=int(index),
        m1_rms_px=m1_rms if "M1" in active else 0.0,
        m1_seed=m1_seed,
        theta_deg=theta if "P1" in active else 0.0,
        phase_dx_vox=pdx if "P2" in active else 0.0,
        phase_dy_vox=pdy if "P2" in active else 0.0,
        mis_dir_deg=mis_dir,
        mis_len_vox=mis_len if "P3" in active else 0.0,
        s_star_mm=s_star if "P4" in active else 0.0,
        sigma_b_vox=sb if "P5" in active else 0.0,
        dose_f=dose if "P6" in active else 1.0,
        noise_seed=noise_seed,
        offset_hu=offset if "P7" in active else 0.0,
    )
    if arm == "S1":
        d["dz_mm"] = dz
    elif arm == "S2":
        d["motion_len_mm"], d["motion_dir_deg"] = mot_len, mot_dir
    elif arm == "S3":
        d["m2"] = True
    elif arm == "S4":
        d["m3"] = True
    elif arm == "S5":
        d["dose_f"] = dose_low
    elif arm == "S6":
        # P4 replaced by a draw up to 0.90 mm, coarser than any COCA spacing.
        if native_spacing_mm < S6_MAX_SPACING_MM:
            d["s_star_mm"] = native_spacing_mm + s6_u * (S6_MAX_SPACING_MM - native_spacing_mm)
        else:
            d["s_star_mm"] = 0.0
    return Draw(**d)


def arm_draw_plan(arms: list[str]) -> list[tuple[str, int]]:
    """(arm, index) pairs for the requested arms, in a stable order."""
    plan: list[tuple[str, int]] = []
    for arm in arms:
        if arm == "primary":
            n = N_PRIMARY_DRAWS
        elif arm in SECONDARY_FACTORS:
            n = N_SECONDARY_DRAWS
        elif arm.startswith("attr_") and arm[5:] in PRIMARY_FACTORS:
            n = N_ATTRIBUTION_DRAWS
        else:
            raise ValueError(f"unknown arm {arm!r}")
        plan.extend((arm, i) for i in range(n))
    return plan


def expand_arms(spec: str) -> list[str]:
    """'primary', 'secondary', 'attribution', 'all' or a comma list of arm names."""
    out: list[str] = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok == "all":
            out += ["primary", *SECONDARY_FACTORS, *[f"attr_{f}" for f in PRIMARY_FACTORS]]
        elif tok == "secondary":
            out += list(SECONDARY_FACTORS)
        elif tok == "attribution":
            out += [f"attr_{f}" for f in PRIMARY_FACTORS]
        else:
            out.append(tok)
    seen, uniq = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


# ───────────────────────── M1: polygon jitter ─────────────────────────


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def jitter_polygon(points: np.ndarray, rms_px: float, rng: np.random.Generator):
    """Displace a closed polygon along its outward vertex normals.

    Field = a0 + sum_{k=1..3} (a_k cos 2 pi k t + b_k sin 2 pi k t) over
    arc-length t, coefficients N(0, 1), scaled to RMS ``rms_px``, each vertex
    capped at M1_CAP_PX. The a0 term is a systematic dilation or erosion. If
    the area falls below M1_AREA_FLOOR of the original, the field is halved
    until it does not (after 20 halvings the polygon is left unchanged).

    Always consumes 7 normals from ``rng`` so the stream does not depend on
    the floor loop. Returns ``(new_points, area_ratio)``.
    """
    coefs = rng.standard_normal(1 + 2 * M1_FOURIER_ORDER)
    p = np.asarray(points, dtype=float)
    n = len(p)
    if n < 3 or rms_px <= 0.0:
        return p, 1.0
    seg = np.roll(p, -1, axis=0) - p
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    perim = float(seg_len.sum())
    area0 = abs(_signed_area(p))
    if perim == 0.0 or area0 == 0.0:
        return p, 1.0
    t = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]]) / perim
    g = np.full(n, coefs[0])
    for k in range(1, M1_FOURIER_ORDER + 1):
        g += coefs[2 * k - 1] * np.cos(2 * np.pi * k * t) + coefs[2 * k] * np.sin(2 * np.pi * k * t)
    rms = float(np.sqrt(np.mean(g ** 2)))
    if rms == 0.0:
        return p, 1.0
    delta = np.clip(rms_px * g / rms, -M1_CAP_PX, M1_CAP_PX)

    tang = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    tlen = np.hypot(tang[:, 0], tang[:, 1])
    safe = np.where(tlen > 0, tlen, 1.0)
    normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1) / safe[:, None]
    normal[tlen == 0] = 0.0
    if _signed_area(p) < 0:
        normal = -normal

    scale = 1.0
    for _ in range(20):
        q = p + (scale * delta)[:, None] * normal
        ratio = abs(_signed_area(q)) / area0
        if ratio >= M1_AREA_FLOOR:
            return q, ratio
        scale *= 0.5
    return p, 1.0


def jitter_parse_result(
    parse_result: ParseResult,
    rms_px: float,
    seed: int,
    excluded_roi_ids: set,
) -> ParseResult:
    """M1 on every rasterisable ROI. ``rms_px == 0`` returns the input object.

    Area is scaled by the polygon's own area ratio (XML Area x ratio), so a
    zero field reproduces the XML exactly. XML Max / Mean stay frozen (D016).
    Excluded (D002), dirty and < 3-point ROIs are left untouched.
    """
    if rms_px <= 0.0:
        return parse_result
    rng = np.random.default_rng(int(seed))
    new_slices = []
    for sl in parse_result.slices:
        new_rois = []
        for roi_idx, roi in enumerate(sl.rois):
            if ((sl.image_index, roi_idx) in excluded_roi_ids or roi.vessel is None
                    or roi.n_points < 3 or len(roi.points_px) < 3):
                new_rois.append(roi)
                continue
            q, ratio = jitter_polygon(np.asarray(roi.points_px, float), rms_px, rng)
            new_rois.append(replace(
                roi,
                points_px=tuple((float(x), float(y)) for x, y in q),
                area_cm2=roi.area_cm2 * ratio,
            ))
        new_slices.append(replace(sl, rois=tuple(new_rois)))
    return replace(parse_result, slices=tuple(new_slices))


# ───────────────────────── image-side factors ─────────────────────────


def _img(arr: np.ndarray, spacing) -> sitk.Image:
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    im.SetSpacing(tuple(float(s) for s in spacing))
    return im


def spatial_transform(ct: np.ndarray, mask: np.ndarray, spacing, draw: Draw,
                      background_hu: float) -> tuple[np.ndarray, np.ndarray]:
    """P1 + P2 (+ S1) on CT and mask together, P3 on the CT only, each image
    resampled once. Content moves as x' = R(x - c) + c + t."""
    from predict.stability.perturbations import _volume_center_physical
    sx, sy = float(spacing[0]), float(spacing[1])
    t_phase = (draw.phase_dx_vox * sx, draw.phase_dy_vox * sy, draw.dz_mm)
    rad = math.radians(draw.mis_dir_deg)
    t_mis = (draw.mis_len_vox * sx * math.cos(rad), draw.mis_len_vox * sy * math.sin(rad), 0.0)
    moves_mask = draw.theta_deg != 0.0 or any(v != 0.0 for v in t_phase)
    moves_ct = moves_mask or draw.mis_len_vox != 0.0
    if not moves_ct:
        return ct, mask

    ct_img = _img(ct.astype(np.float32), spacing)
    centre = _volume_center_physical(ct_img)

    def inverse(t):
        tr = sitk.Euler3DTransform()
        tr.SetCenter(centre)
        tr.SetRotation(0.0, 0.0, math.radians(draw.theta_deg))
        tr.SetTranslation(tuple(float(v) for v in t))
        return tr.GetInverse()

    t_ct = tuple(a + b for a, b in zip(t_phase, t_mis))
    ct_out = sitk.GetArrayFromImage(sitk.Resample(
        ct_img, ct_img, inverse(t_ct), sitk.sitkLinear, float(background_hu), sitk.sitkFloat32))
    if not moves_mask:
        return ct_out, mask
    m_img = _img(mask.astype(np.float32), spacing)
    m_out = sitk.GetArrayFromImage(sitk.Resample(
        m_img, m_img, inverse(t_phase), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
    return ct_out, (m_out >= 0.5).astype(np.uint8)


def resolution_loss(ct: np.ndarray, mask: np.ndarray, spacing, s_star_mm: float,
                    background_hu: float) -> tuple[np.ndarray, np.ndarray]:
    """P4 / S6: in-plane down to ``s_star_mm`` and back (CT B-spline, mask
    nearest, the D005 interpolators). ``s_star_mm == 0`` is a no-op."""
    if s_star_mm <= 0.0:
        return ct, mask
    ref_ct = _img(ct.astype(np.float32), spacing)
    ref_m = _img(mask.astype(np.uint8), spacing)
    size = ref_ct.GetSize()
    coarse_sp = (s_star_mm, s_star_mm, float(spacing[2]))
    coarse_size = [max(1, int(round(size[0] * spacing[0] / s_star_mm))),
                   max(1, int(round(size[1] * spacing[1] / s_star_mm))), size[2]]

    def down(im, interp, fill):
        r = sitk.ResampleImageFilter()
        r.SetOutputSpacing(coarse_sp)
        r.SetSize(coarse_size)
        r.SetOutputOrigin(im.GetOrigin())
        r.SetOutputDirection(im.GetDirection())
        r.SetTransform(sitk.Transform())
        r.SetInterpolator(interp)
        r.SetDefaultPixelValue(fill)
        return r.Execute(im)

    def up(im, ref, interp, fill):
        return sitk.Resample(im, ref, sitk.Transform(), interp, fill, im.GetPixelID())

    ct_out = up(down(ref_ct, sitk.sitkBSpline, background_hu), ref_ct, sitk.sitkBSpline, background_hu)
    m_out = up(down(ref_m, sitk.sitkNearestNeighbor, 0), ref_m, sitk.sitkNearestNeighbor, 0)
    return sitk.GetArrayFromImage(ct_out), sitk.GetArrayFromImage(m_out).astype(np.uint8)


def count_vanished_components(before: np.ndarray, after: np.ndarray) -> int:
    """3D components of ``before`` with no voxel left in ``after``."""
    from scipy.ndimage import label
    lab, n = label(before > 0)
    if n == 0:
        return 0
    kept = np.unique(lab[(after > 0) & (lab > 0)])
    return int(n - kept.size)


def motion_kernel(length_vox: float, direction_deg: float) -> np.ndarray | None:
    """S2: normalised in-plane line kernel, or None if shorter than 1 voxel."""
    if length_vox < 1.0:
        return None
    half = int(math.ceil(length_vox / 2.0)) + 1
    k = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.float64)
    rad = math.radians(direction_deg)
    for s in np.linspace(-length_vox / 2.0, length_vox / 2.0, max(2, int(math.ceil(length_vox * 4)))):
        x, y = half + s * math.cos(rad), half + s * math.sin(rad)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        k[y0, x0] += (1 - fx) * (1 - fy)
        k[y0, x0 + 1] += fx * (1 - fy)
        k[y0 + 1, x0] += (1 - fx) * fy
        k[y0 + 1, x0 + 1] += fx * fy
    return k / k.sum()


def correlated_noise(shape, sd: float, corr_sigma_vox: float, seed: int) -> np.ndarray:
    """P6: white Gaussian noise, in-plane Gaussian-filtered, rescaled to ``sd``."""
    from scipy.ndimage import gaussian_filter
    white = np.random.default_rng(int(seed)).standard_normal(shape)
    field = gaussian_filter(white, sigma=(0.0, corr_sigma_vox, corr_sigma_vox)) if corr_sigma_vox > 0 else white
    s = float(field.std())
    return field * (sd / s) if s > 0 else field


def apply_m2(ct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """S3: voxels >= 130 HU inside a 1-voxel in-plane dilation of the mask,
    keeping only 3D components that overlap the mask."""
    from scipy.ndimage import binary_dilation, label
    st = np.zeros((3, 3, 3), bool)
    st[1] = True
    cand = binary_dilation(mask > 0, structure=st) & (ct >= CALCIUM_HU)
    lab, n = label(cand)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    keep = np.unique(lab[(mask > 0) & (lab > 0)])
    return np.isin(lab, keep[keep > 0]).astype(np.uint8)


def apply_m3(mask: np.ndarray, spacing) -> np.ndarray:
    """S4: drop per-slice 8-connected components smaller than 1 mm^2."""
    from scipy.ndimage import label
    px_area = float(spacing[0]) * float(spacing[1])
    out = np.zeros_like(mask, dtype=np.uint8)
    for z in range(mask.shape[0]):
        lab, n = label(mask[z] > 0, structure=np.ones((3, 3), bool))
        if n == 0:
            continue
        counts = np.bincount(lab.ravel())
        big = np.flatnonzero(counts * px_area >= AGATSTON_MIN_AREA_MM2)
        big = big[big > 0]
        out[z] = np.isin(lab, big)
    return out


@dataclass(frozen=True)
class VolumeResult:
    ct: np.ndarray          # int16, clipped
    mask: np.ndarray        # uint8
    n_vanished: int         # components lost to P4 / S6


def apply_draw_to_volumes(
    ct: np.ndarray,
    mask: np.ndarray,
    spacing,
    draw: Draw,
    *,
    sigma0: float,
    corr_sigma_vox: float,
    background_hu: float,
    clip_min: float,
    clip_max: float,
) -> VolumeResult:
    """Everything after M1, in D033 order. A zero draw returns the inputs."""
    from scipy.ndimage import gaussian_filter
    ct_w, m_w = spatial_transform(ct, mask, spacing, draw, background_hu)

    n_vanished = 0
    if draw.s_star_mm > 0.0:
        m_before = m_w
        ct_w, m_w = resolution_loss(ct_w, m_w, spacing, draw.s_star_mm, background_hu)
        n_vanished = count_vanished_components(m_before, m_w)

    changed = ct_w is not ct
    if draw.sigma_b_vox > 0.0:
        ct_w = gaussian_filter(np.asarray(ct_w, np.float32), sigma=(0.0, draw.sigma_b_vox, draw.sigma_b_vox))
        changed = True
    if draw.motion_len_mm > 0.0:
        import cv2
        k = motion_kernel(draw.motion_len_mm / float(spacing[0]), draw.motion_dir_deg)
        if k is not None:
            f = np.asarray(ct_w, np.float32)
            ct_w = np.stack([cv2.filter2D(s, -1, k.astype(np.float32), borderType=cv2.BORDER_REPLICATE) for s in f])
            changed = True
    if draw.dose_f < 1.0:
        sd = sigma0 * math.sqrt(1.0 / draw.dose_f - 1.0)
        if np.isfinite(sd) and sd > 0:
            ct_w = np.asarray(ct_w, np.float32) + correlated_noise(ct.shape, sd, corr_sigma_vox, draw.noise_seed).astype(np.float32)
            changed = True
    if draw.offset_hu != 0.0:
        ct_w = np.asarray(ct_w, np.float32) + np.float32(draw.offset_hu)
        changed = True

    if changed:
        ct_out = np.rint(np.clip(ct_w, clip_min, clip_max)).astype(np.int16)
    else:
        ct_out = ct
    m_out = m_w
    if draw.m2:
        m_out = apply_m2(ct_out, m_out)
    if draw.m3:
        m_out = apply_m3(m_out, spacing)
    return VolumeResult(ct=ct_out, mask=np.asarray(m_out, dtype=np.uint8), n_vanished=n_vanished)


__all__ = [
    "PRIMARY_FACTORS", "SECONDARY_FACTORS",
    "N_PRIMARY_DRAWS", "N_SECONDARY_DRAWS", "N_ATTRIBUTION_DRAWS",
    "Calibration", "Draw", "VolumeResult",
    "immerkaer_sigma", "noise_lag1_correlation", "lag1_of_filtered_white_noise", "sigma_from_lag1",
    "baseline_draw", "sample_draw", "arm_draw_plan", "expand_arms",
    "jitter_polygon", "jitter_parse_result",
    "spatial_transform", "resolution_loss", "count_vanished_components",
    "motion_kernel", "correlated_noise", "apply_m2", "apply_m3",
    "apply_draw_to_volumes",
]
