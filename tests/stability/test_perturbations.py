"""Tests for predict.stability.perturbations.

Verifies the D014 design (mask held fixed):

- 14 deterministic perturbation specs in fixed order.
- Rotation and translation transform the CT only; the mask is never returned
  from rotate / translate (callers pass the unchanged mask through to the
  extractor).
- Noise is deterministic per (pid, sigma).
- All transforms preserve volume geometry metadata (size, spacing, origin,
  direction).
"""
from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from predict.config import load_config
from predict.stability.perturbations import (
    PerturbationSpec,
    add_gaussian_noise,
    apply_perturbation,
    enumerate_perturbations,
    noise_seed,
    rotate,
    translate,
)


# ─────────────────────── helpers ───────────────────────


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _make_test_ct(
    shape: tuple[int, int, int] = (5, 32, 32),
    spacing: tuple[float, float, float] = (0.5, 0.5, 3.0),
) -> sitk.Image:
    """Build a synthetic CT with a high-HU square in the middle of every slice."""
    arr = np.full(shape, -1000.0, dtype=np.float32)
    arr[:, 10:14, 8:12] = 300.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def _arr(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


# ─────────────────────── enumerate_perturbations ───────────────────────


def test_enumerate_returns_14_specs(cfg):
    specs = enumerate_perturbations(cfg)
    assert len(specs) == 14


def test_enumerate_names_match_locked_set(cfg):
    expected = {
        "rotate_+5", "rotate_-5", "rotate_+10", "rotate_-10",
        "translate_+2_x", "translate_-2_x", "translate_+5_x", "translate_-5_x",
        "translate_+2_y", "translate_-2_y", "translate_+5_y", "translate_-5_y",
        "noise_5", "noise_10",
    }
    names = {s.name for s in enumerate_perturbations(cfg)}
    assert names == expected


def test_enumerate_order_is_deterministic(cfg):
    a = [s.name for s in enumerate_perturbations(cfg)]
    b = [s.name for s in enumerate_perturbations(cfg)]
    assert a == b


def test_enumerate_specs_are_frozen(cfg):
    specs = enumerate_perturbations(cfg)
    with pytest.raises(AttributeError):
        specs[0].name = "modified"  # type: ignore[misc]


# ─────────────────────── rotation ───────────────────────


def test_rotate_preserves_geometry(cfg):
    ct = _make_test_ct()
    rotated = rotate(ct, 5.0, background_hu=cfg.stability.background_fill_hu)
    assert rotated.GetSize() == ct.GetSize()
    assert rotated.GetSpacing() == ct.GetSpacing()
    assert rotated.GetOrigin() == ct.GetOrigin()
    assert rotated.GetDirection() == ct.GetDirection()


def test_rotate_zero_degrees_is_identity(cfg):
    ct = _make_test_ct()
    rotated = rotate(ct, 0.0, background_hu=cfg.stability.background_fill_hu)
    np.testing.assert_allclose(_arr(rotated), _arr(ct), atol=1e-6)


def test_rotate_360_degrees_is_near_identity(cfg):
    ct = _make_test_ct()
    rotated = rotate(ct, 360.0, background_hu=cfg.stability.background_fill_hu)
    # Linear interpolation introduces small error after a full revolution.
    np.testing.assert_allclose(_arr(rotated), _arr(ct), atol=1e-3)


def test_rotate_actually_changes_voxels_for_nonzero_angle(cfg):
    ct = _make_test_ct()
    rotated = rotate(ct, 10.0, background_hu=cfg.stability.background_fill_hu)
    assert not np.allclose(_arr(rotated), _arr(ct))


def test_rotate_uses_background_fill_outside_volume(cfg):
    """A 30-degree rotation of a square zero-filled CT must produce corners
    at the background HU.

    Setup choice: interior is filled with 0.0 (not -1000) so we can cleanly
    distinguish "background fill applied" from "edge value sampled". A
    30-degree rotation on a 16x16 plane maps the four output corners to
    input coordinates outside the FOV (distance ~10.6 from centre, after
    inverse rotation the x or y coordinate is negative).
    """
    arr = np.zeros((1, 16, 16), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 3.0))
    rotated = rotate(img, 30.0, background_hu=cfg.stability.background_fill_hu)
    out = sitk.GetArrayFromImage(rotated)
    bg = cfg.stability.background_fill_hu
    assert out[0, 0, 0]   == pytest.approx(bg, abs=1.0)
    assert out[0, 0, -1]  == pytest.approx(bg, abs=1.0)
    assert out[0, -1, 0]  == pytest.approx(bg, abs=1.0)
    assert out[0, -1, -1] == pytest.approx(bg, abs=1.0)


# ─────────────────────── translation ───────────────────────


def test_translate_preserves_geometry(cfg):
    ct = _make_test_ct()
    shifted = translate(ct, 2.0, 0.0,
                        background_hu=cfg.stability.background_fill_hu)
    assert shifted.GetSize() == ct.GetSize()
    assert shifted.GetSpacing() == ct.GetSpacing()
    assert shifted.GetOrigin() == ct.GetOrigin()
    assert shifted.GetDirection() == ct.GetDirection()


def test_translate_zero_is_identity(cfg):
    ct = _make_test_ct()
    shifted = translate(ct, 0.0, 0.0,
                        background_hu=cfg.stability.background_fill_hu)
    np.testing.assert_allclose(_arr(shifted), _arr(ct), atol=1e-6)


def test_translate_x_moves_high_hu_block(cfg):
    """Shifting the CT by +5 mm in x with spacing 0.5 mm should move the
    high-HU block by ~10 voxels in the +x direction.

    NOTE: SimpleITK's TranslationTransform with positive (tx, ty, tz)
    translates the OUTPUT coordinate by (tx, ty, tz) before mapping back to
    input via the inverse. The net visual effect on the resampled image is
    a shift of the content in the OPPOSITE direction. We test that the
    centroid moves in some consistent direction by a sensible magnitude, not
    a specific sign, to keep the test robust to convention quirks.
    """
    ct = _make_test_ct()
    shifted = translate(ct, 5.0, 0.0,
                        background_hu=cfg.stability.background_fill_hu)

    def _high_hu_centroid_xy(img: sitk.Image) -> tuple[float, float]:
        arr = _arr(img)
        # Centroid in voxel coords of voxels above 100 HU (the block).
        coords = np.argwhere(arr > 100.0)
        if coords.size == 0:
            return (np.nan, np.nan)
        z, y, x = coords.mean(axis=0)
        sx, sy, _ = img.GetSpacing()
        return (float(x * sx), float(y * sy))

    before_x, before_y = _high_hu_centroid_xy(ct)
    after_x, after_y = _high_hu_centroid_xy(shifted)

    assert abs(after_x - before_x) > 4.0   # ≈5 mm shift in x
    assert abs(after_y - before_y) < 0.5   # negligible shift in y


def test_translate_y_moves_high_hu_block(cfg):
    ct = _make_test_ct()
    shifted = translate(ct, 0.0, 2.0,
                        background_hu=cfg.stability.background_fill_hu)

    def _high_hu_centroid_xy(img: sitk.Image) -> tuple[float, float]:
        arr = _arr(img)
        coords = np.argwhere(arr > 100.0)
        z, y, x = coords.mean(axis=0)
        sx, sy, _ = img.GetSpacing()
        return (float(x * sx), float(y * sy))

    before_x, before_y = _high_hu_centroid_xy(ct)
    after_x, after_y = _high_hu_centroid_xy(shifted)

    assert abs(after_y - before_y) > 1.5
    assert abs(after_x - before_x) < 0.5


# ─────────────────────── noise ───────────────────────


def test_noise_deterministic_with_same_seed():
    ct = _make_test_ct()
    a = add_gaussian_noise(ct, 5.0, seed=7)
    b = add_gaussian_noise(ct, 5.0, seed=7)
    np.testing.assert_allclose(_arr(a), _arr(b), atol=1e-12)


def test_noise_different_seed_changes_pattern():
    ct = _make_test_ct()
    a = add_gaussian_noise(ct, 5.0, seed=7)
    b = add_gaussian_noise(ct, 5.0, seed=8)
    assert not np.allclose(_arr(a), _arr(b))


def test_noise_std_matches_sigma_on_constant_image():
    arr = np.full((50, 50, 50), 0.0, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 3.0))
    noisy = add_gaussian_noise(img, sigma_hu=10.0, seed=42,
                               clip_min=-200.0, clip_max=3000.0)
    measured = float(_arr(noisy).std())
    assert abs(measured - 10.0) < 0.3


def test_noise_clips_to_configured_range():
    arr = np.full((4, 4, 4), 2995.0, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    # Big noise: many samples will exceed clip_max=3000.
    noisy = add_gaussian_noise(img, sigma_hu=200.0, seed=1,
                               clip_min=-200.0, clip_max=3000.0)
    arr_out = _arr(noisy)
    assert float(arr_out.max()) <= 3000.0
    assert float(arr_out.min()) >= -200.0


def test_noise_preserves_geometry():
    ct = _make_test_ct()
    noisy = add_gaussian_noise(ct, 5.0, seed=1)
    assert noisy.GetSize() == ct.GetSize()
    assert noisy.GetSpacing() == ct.GetSpacing()
    assert noisy.GetOrigin() == ct.GetOrigin()
    assert noisy.GetDirection() == ct.GetDirection()


# ─────────────────────── noise_seed ───────────────────────


def test_noise_seed_deterministic_per_pid_and_sigma():
    s1 = noise_seed("306", 10.0, multiplier=1000)
    s2 = noise_seed("306", 10.0, multiplier=1000)
    assert s1 == s2


def test_noise_seed_differs_across_patients():
    s_a = noise_seed("306", 10.0, multiplier=1000)
    s_b = noise_seed("307", 10.0, multiplier=1000)
    assert s_a != s_b


def test_noise_seed_differs_across_sigmas():
    s5 = noise_seed("306", 5.0, multiplier=1000)
    s10 = noise_seed("306", 10.0, multiplier=1000)
    assert s5 != s10


def test_noise_seed_handles_non_numeric_pid():
    s = noise_seed("abc", 5.0, multiplier=1000)
    assert isinstance(s, int)


# ─────────────────────── apply_perturbation ───────────────────────


def test_apply_perturbation_dispatches_rotation(cfg):
    ct = _make_test_ct()
    spec = PerturbationSpec(name="rotate_+5", kind="rotate", rotation_deg=5.0)
    out = apply_perturbation(ct, spec, pid="1", cfg=cfg)
    assert out.GetSize() == ct.GetSize()


def test_apply_perturbation_dispatches_translation(cfg):
    ct = _make_test_ct()
    spec = PerturbationSpec(name="translate_+2_x", kind="translate", tx_mm=2.0)
    out = apply_perturbation(ct, spec, pid="1", cfg=cfg)
    assert out.GetSize() == ct.GetSize()


def test_apply_perturbation_dispatches_noise(cfg):
    ct = _make_test_ct()
    spec = PerturbationSpec(name="noise_5", kind="noise", sigma_hu=5.0)
    out = apply_perturbation(ct, spec, pid="306", cfg=cfg)
    assert not np.allclose(_arr(out), _arr(ct))


def test_apply_perturbation_unknown_kind_raises(cfg):
    ct = _make_test_ct()
    bogus = PerturbationSpec(name="bogus", kind="rotate")  # type: ignore[arg-type]
    object.__setattr__(bogus, "kind", "bogus")
    with pytest.raises(ValueError, match="Unknown perturbation kind"):
        apply_perturbation(ct, bogus, pid="1", cfg=cfg)


def test_apply_all_14_perturbations_on_synthetic_ct(cfg):
    ct = _make_test_ct()
    for spec in enumerate_perturbations(cfg):
        out = apply_perturbation(ct, spec, pid="1", cfg=cfg)
        assert out.GetSize() == ct.GetSize()
        assert out.GetSpacing() == ct.GetSpacing()
