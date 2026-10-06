"""Tests for predict.stability.realistic (D033) and icc_1_1.

Synthetic inputs only; no patient data.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation
from predict.stability.icc import icc_1_1
from predict.stability.realistic import (
    M1_AREA_FLOOR,
    M1_CAP_PX,
    N_PRIMARY_DRAWS,
    PRIMARY_FACTORS,
    Calibration,
    Draw,
    apply_draw_to_volumes,
    apply_m2,
    apply_m3,
    arm_draw_plan,
    correlated_noise,
    expand_arms,
    immerkaer_sigma,
    jitter_parse_result,
    jitter_polygon,
    lag1_of_filtered_white_noise,
    resolution_loss,
    sample_draw,
    sigma_from_lag1,
)

SP = (0.375, 0.375, 3.0)
CALIB = Calibration(sigma0={"7": 20.0}, sigma_n={"K": 0.8}, sigma_n_default=0.8,
                    sigma_delta=0.4, native_spacings=(0.30, 0.38, 0.45, 0.60))


# ───────────────────────── ICC(1,1) ─────────────────────────


def test_icc_1_1_hand_computed():
    # subject means 1.5, 3.5, 5.5; MSB = 8, MSW = 0.5 -> (8 - 0.5) / (8 + 0.5)
    m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    icc, lo, hi = icc_1_1(m)
    assert icc == pytest.approx(7.5 / 8.5, abs=1e-12)
    assert lo <= icc <= hi


def test_icc_1_1_perfect_agreement_is_one():
    m = np.column_stack([np.arange(10.0)] * 5)
    assert icc_1_1(m)[0] == 1.0


def test_icc_1_1_pure_noise_near_zero():
    rng = np.random.default_rng(0)
    assert abs(icc_1_1(rng.standard_normal((400, 20)))[0]) < 0.05


def test_icc_1_1_listwise_deletion():
    m = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0]])
    assert icc_1_1(m)[0] == pytest.approx(icc_1_1(m[[0, 2, 3]])[0])


# ───────────────────────── draws ─────────────────────────


def test_primary_seed_matches_d033_text():
    d = sample_draw("7", "primary", 3, CALIB, 0.38)
    rng = np.random.default_rng([7, 3, 33])
    assert d.m1_rms_px == pytest.approx(rng.uniform(0.0, 0.5))


def test_draws_are_reproducible_and_distinct():
    a = sample_draw("7", "primary", 0, CALIB, 0.38)
    assert a == sample_draw("7", "primary", 0, CALIB, 0.38)
    assert a != sample_draw("7", "primary", 1, CALIB, 0.38)
    assert a != sample_draw("8", "primary", 0, CALIB, 0.38)


def test_attribution_draw_activates_one_factor():
    d = sample_draw("7", "attr_P3", 0, CALIB, 0.38)
    assert d.mis_len_vox > 0
    assert d.m1_rms_px == 0 and d.theta_deg == 0 and d.phase_dx_vox == 0
    assert d.s_star_mm == 0 and d.sigma_b_vox == 0 and d.dose_f == 1.0 and d.offset_hu == 0


def test_p4_never_sharpens():
    for i in range(50):
        d = sample_draw("7", "primary", i, CALIB, 0.45)
        assert d.s_star_mm == 0.0 or d.s_star_mm > 0.45


def test_s6_range():
    for i in range(20):
        d = sample_draw("7", "S6", i, CALIB, 0.38)
        assert 0.38 <= d.s_star_mm <= 0.90


def test_primary_magnitudes_within_d033_bounds():
    for i in range(50):
        d = sample_draw("7", "primary", i, CALIB, 0.30)
        assert 0 <= d.m1_rms_px <= 0.5 and abs(d.theta_deg) <= 5
        assert abs(d.phase_dx_vox) <= 0.5 and abs(d.phase_dy_vox) <= 0.5
        assert 0 <= d.mis_len_vox <= 1 and 0.5 <= d.dose_f <= 1 and abs(d.offset_hu) <= 5
        assert 0 <= d.sigma_b_vox <= CALIB.sigma_delta
        assert d.dz_mm == 0 and d.motion_len_mm == 0 and not d.m2 and not d.m3


def test_arm_plan_counts():
    assert len(arm_draw_plan(["primary"])) == N_PRIMARY_DRAWS
    assert len(arm_draw_plan(expand_arms("all"))) == 20 + 6 * 10 + len(PRIMARY_FACTORS) * 10


# ───────────────────────── M1 ─────────────────────────


def _circle(r=5.0, n=12, cx=100.0, cy=80.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


@pytest.mark.parametrize("r,n", [(5.0, 12), (1.2, 4), (0.8, 3)])
def test_jitter_polygon_bounds(r, n):
    p = _circle(r, n)
    for seed in range(200):
        q, ratio = jitter_polygon(p, 0.5, np.random.default_rng(seed))
        assert np.max(np.hypot(*(q - p).T)) <= M1_CAP_PX + 1e-9
        assert ratio >= M1_AREA_FLOOR - 1e-12
        assert ratio == pytest.approx(_area(q) / _area(p))


def test_jitter_polygon_zero_is_identity():
    p = _circle()
    q, ratio = jitter_polygon(p, 0.0, np.random.default_rng(0))
    assert np.array_equal(q, p) and ratio == 1.0


def _roi(points, vessel="LAD", area=0.1):
    pts = tuple((float(x), float(y)) for x, y in points)
    return ROI(vessel_raw="x", vessel=vessel, area_cm2=area, mean_hu=300.0, max_hu=500.0,
               min_hu=130.0, total_hu=0.0, n_points=len(pts), points_px=pts, points_mm=(),
               center_xyz=(0.0, 0.0, 0.0))


def _parse():
    return ParseResult(pid="7", slices=(SliceAnnotation(image_index=3, rois=(
        _roi(_circle()), _roi(_circle(cx=50.0)), _roi(_circle(cx=20.0), vessel=None))),))


def test_jitter_parse_result_zero_returns_same_object():
    pr = _parse()
    assert jitter_parse_result(pr, 0.0, 1, set()) is pr


def test_jitter_parse_result_respects_exclusions_and_dirty():
    pr = _parse()
    out = jitter_parse_result(pr, 0.5, 1, {(3, 1)})
    r0, r1, r2 = out.slices[0].rois
    assert r0.points_px != pr.slices[0].rois[0].points_px
    assert r1 == pr.slices[0].rois[1]      # excluded (D002)
    assert r2 == pr.slices[0].rois[2]      # dirty
    ratio = _area(np.asarray(r0.points_px)) / _area(np.asarray(pr.slices[0].rois[0].points_px))
    assert r0.area_cm2 == pytest.approx(0.1 * ratio)
    assert r0.max_hu == 500.0 and r0.mean_hu == 300.0   # XML HU frozen (D016)


# ───────────────────────── volumes ─────────────────────────


def _phantom():
    ct = np.full((3, 200, 200), 40, dtype=np.int16)
    ct[:, :5, :] = -200
    mask = np.zeros(ct.shape, np.uint8)
    ct[1, 60:68, 130:138] = 800
    mask[1, 60:68, 130:138] = 1
    return ct, mask


def _centroid(a):
    z, y, x = np.nonzero(a)
    return np.array([x.mean(), y.mean()])


def _kw():
    return dict(sigma0=20.0, corr_sigma_vox=0.8, background_hu=-1024.0, clip_min=-200.0, clip_max=3000.0)


def test_zero_draw_is_exact_identity():
    ct, mask = _phantom()
    vr = apply_draw_to_volumes(ct, mask, SP, Draw(arm="primary", index=0), **_kw())
    assert vr.ct is ct and vr.mask is mask and vr.n_vanished == 0


def test_p2_moves_ct_and_mask_together():
    ct, mask = _phantom()
    d = Draw(arm="attr_P2", index=0, phase_dx_vox=0.5, phase_dy_vox=-0.5)
    vr = apply_draw_to_volumes(ct, mask, SP, d, **_kw())
    shift_ct = _centroid(vr.ct > 400) - _centroid(ct > 400)
    shift_m = _centroid(vr.mask) - _centroid(mask)
    assert np.allclose(shift_ct, shift_m, atol=0.51)


def test_p1_rotates_ct_and_mask_together():
    ct, mask = _phantom()
    vr = apply_draw_to_volumes(ct, mask, SP, Draw(arm="attr_P1", index=0, theta_deg=5.0), **_kw())
    assert np.allclose(_centroid(vr.ct > 400), _centroid(vr.mask), atol=0.6)
    assert np.linalg.norm(_centroid(vr.mask) - _centroid(mask)) > 2.0


def _weighted_centroid(ct, background=40.0):
    """Intensity-weighted centroid of the blob. Linear interpolation preserves
    it exactly, unlike a thresholded centroid, which snaps a sub-voxel shift
    to whole voxels (a 0.707-voxel shift reads as 1.0 per axis)."""
    w = np.clip(ct.astype(np.float64) - background, 0.0, None)
    z, y, x = np.indices(ct.shape)
    return np.array([(w * x).sum(), (w * y).sum()]) / w.sum()


def test_p3_moves_only_ct_by_at_most_one_voxel():
    ct, mask = _phantom()
    d = Draw(arm="attr_P3", index=0, mis_dir_deg=45.0, mis_len_vox=1.0)
    vr = apply_draw_to_volumes(ct, mask, SP, d, **_kw())
    assert np.array_equal(vr.mask, mask)
    shift = _weighted_centroid(vr.ct) - _weighted_centroid(ct)
    expected = np.array([math.cos(math.radians(45.0)), math.sin(math.radians(45.0))])
    assert np.allclose(shift, expected, atol=0.02)
    assert np.linalg.norm(shift) <= 1.0 + 0.02


def test_resolution_loss_preserves_shape_and_noop():
    ct, mask = _phantom()
    c0, m0 = resolution_loss(ct, mask, SP, 0.0, -1024.0)
    assert c0 is ct and m0 is mask
    c1, m1 = resolution_loss(ct, mask, SP, 0.7, -1024.0)
    assert c1.shape == ct.shape and m1.shape == mask.shape


def test_correlated_noise_sd_and_correlation():
    f = correlated_noise((4, 256, 256), 12.0, 1.0, seed=3)
    assert f.std() == pytest.approx(12.0, rel=1e-6)
    a, b = f[:, :, :-1].ravel(), f[:, :, 1:].ravel()
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(lag1_of_filtered_white_noise(1.0), abs=0.02)


def test_sigma_from_lag1_inverts():
    for s in (0.6, 1.0, 1.3, 2.5):
        assert sigma_from_lag1(lag1_of_filtered_white_noise(s)) == pytest.approx(s, abs=1e-3)
    assert sigma_from_lag1(0.0) == 0.0


def test_lag1_matches_continuous_limit_at_large_sigma():
    assert lag1_of_filtered_white_noise(3.0) == pytest.approx(math.exp(-1 / 36), abs=1e-3)


def test_immerkaer_recovers_white_noise_sd():
    rng = np.random.default_rng(1)
    vol = 50.0 + 20.0 * rng.standard_normal((4, 200, 200))
    assert immerkaer_sigma(vol) == pytest.approx(20.0, rel=0.08)


def test_m3_drops_sub_mm2_components():
    m = np.zeros((1, 40, 40), np.uint8)
    m[0, 2:4, 2:3] = 1           # 2 voxels = 0.28 mm^2
    m[0, 20:24, 20:24] = 1       # 16 voxels = 2.25 mm^2
    out = apply_m3(m, SP)
    assert out[0, 2:4, 2].sum() == 0 and out[0, 20:24, 20:24].sum() == 16


def test_m2_threshold_within_dilation_and_overlap():
    ct = np.full((1, 40, 40), 40, np.int16)
    m = np.zeros(ct.shape, np.uint8)
    m[0, 10:14, 10:14] = 1
    ct[0, 9:15, 9:15] = 400      # bright one voxel beyond the mask: kept via dilation
    ct[0, 30:33, 30:33] = 400    # separate bright blob: dropped
    ct[0, 10, 10] = 100          # below 130 inside the mask: dropped
    out = apply_m2(ct, m)
    assert out[0, 9:15, 9:15].sum() == 36 - 1
    assert out[0, 30:33, 30:33].sum() == 0
