"""Tests for predict.preprocess.slice_matcher."""
from __future__ import annotations

from dataclasses import replace

from predict.io.xml_parser import ROI
from predict.preprocess.slice_matcher import (
    fallback_image_index_to_slice,
    match_roi_to_slice,
    resolve_roi_slice,
)


def _roi_with_center(z: float) -> ROI:
    return ROI(
        vessel_raw="Left Anterior Descending Artery",
        vessel="LAD",
        area_cm2=0.05,
        mean_hu=200.0,
        max_hu=250.0,
        min_hu=150.0,
        total_hu=1000.0,
        n_points=4,
        points_px=((0.0, 0.0),) * 4,
        points_mm=((0.0, 0.0, z),) * 4,
        center_xyz=(0.0, 0.0, z),
    )


def test_match_picks_closest_slice():
    positions = (0.0, 3.0, 6.0, 9.0, 12.0)
    assert match_roi_to_slice(_roi_with_center(6.1), positions, tolerance_mm=1.5) == 2


def test_match_returns_none_when_outside_tolerance():
    positions = (0.0, 3.0, 6.0, 9.0)
    # 7.6 is 1.6 from slice 2 (6.0) and 1.4 from slice 3 (9.0).
    # tolerance_mm = 1.0 makes neither acceptable.
    assert match_roi_to_slice(_roi_with_center(7.6), positions, tolerance_mm=1.0) is None


def test_match_within_tolerance_picks_closer():
    positions = (0.0, 3.0, 6.0, 9.0)
    assert match_roi_to_slice(_roi_with_center(7.6), positions, tolerance_mm=2.0) == 3


def test_match_returns_none_when_no_center():
    positions = (0.0, 3.0, 6.0)
    roi = replace(_roi_with_center(0.0), center_xyz=None)
    assert match_roi_to_slice(roi, positions) is None


def test_match_exact_z():
    positions = (-10.0, -7.0, -4.0, -1.0, 2.0)
    assert match_roi_to_slice(_roi_with_center(-7.0), positions) == 1


def test_fallback_direct_mapping():
    assert fallback_image_index_to_slice(0, 10) == 0
    assert fallback_image_index_to_slice(5, 10) == 5
    assert fallback_image_index_to_slice(9, 10) == 9


def test_fallback_rejects_out_of_bounds():
    assert fallback_image_index_to_slice(-1, 10) is None
    assert fallback_image_index_to_slice(10, 10) is None
    assert fallback_image_index_to_slice(99, 10) is None


# -- resolve_roi_slice: the one rule all callers use --


def test_resolve_uses_center_when_it_matches():
    positions = (0.0, 3.0, 6.0, 9.0)
    assert resolve_roi_slice(_roi_with_center(6.1), 0, positions) == (2, "center")


def test_resolve_never_falls_back_when_center_contradicts_scan():
    """A Center outside the scan is a contradiction, not a missing value:
    ImageIndex 1 is valid but must not be used."""
    positions = (0.0, 3.0, 6.0, 9.0)
    assert resolve_roi_slice(_roi_with_center(40.0), 1, positions) == (None, "center_mismatch")


def test_resolve_falls_back_only_when_center_absent():
    positions = (0.0, 3.0, 6.0)
    roi = replace(_roi_with_center(0.0), center_xyz=None)
    assert resolve_roi_slice(roi, 2, positions) == (2, "image_index_fallback")
    assert resolve_roi_slice(roi, 7, positions) == (None, "unmatched")
