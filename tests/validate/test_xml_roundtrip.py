"""Tests for predict.validate.xml_roundtrip.

The synthetic stamper writes known HU into a polygon, so the round-trip
should pass with delta_mean = 0 exactly.
"""
from __future__ import annotations

from dataclasses import replace

import SimpleITK as sitk

from predict.io.xml_parser import ParseResult, SliceAnnotation
from predict.validate.xml_roundtrip import (
    check_roi,
    failed_roi_ids,
    pass_rate,
    trips_to_rows,
    xml_roundtrip_check,
)

from tests.preprocess._helpers import (
    make_loaded_patient,
    make_parse_result,
    stamp_calcium_square,
)


def _ct_array(loaded):
    return sitk.GetArrayFromImage(loaded.ct_sitk)


def test_roundtrip_passes_on_synthetic_stamp():
    loaded = make_loaded_patient(n_slices=10)
    roi, _ = stamp_calcium_square(loaded, slice_idx=4, hu=300)
    pr = make_parse_result("TEST", image_index=4, roi=roi)

    trips = xml_roundtrip_check(pr, _ct_array(loaded), loaded.slice_positions)
    assert len(trips) == 1
    t = trips[0]
    assert t.passes is True
    assert t.matched_via == "center"
    assert t.matched_slice_idx == 4
    assert t.delta_max == 0.0
    assert abs(t.delta_mean) < 1e-6


def test_roundtrip_fails_on_z_flip_simulation():
    """Stamp on slice 4 but point the centre at slice 7: Max gate must fail."""
    loaded = make_loaded_patient(n_slices=10)
    roi, _ = stamp_calcium_square(loaded, slice_idx=4, hu=300)
    wrong_center = (roi.center_xyz[0], roi.center_xyz[1], loaded.slice_positions[7])
    bad_roi = replace(roi, center_xyz=wrong_center)
    pr = make_parse_result("TEST", image_index=4, roi=bad_roi)

    trips = xml_roundtrip_check(pr, _ct_array(loaded), loaded.slice_positions)
    t = trips[0]
    assert t.passes is False
    assert t.matched_slice_idx == 7
    assert t.delta_max != 0.0  # voxels on slice 7 don't contain the stamped 300 HU
    assert "max_delta" in t.reason


def test_roundtrip_skips_dirty_vessel():
    loaded = make_loaded_patient()
    roi, _ = stamp_calcium_square(loaded, slice_idx=3)
    dirty = replace(roi, vessel=None, vessel_raw="555614876")
    pr = make_parse_result("TEST", image_index=3, roi=dirty)

    trips = xml_roundtrip_check(pr, _ct_array(loaded), loaded.slice_positions)
    assert trips[0].matched_via == "dirty"
    assert trips[0].passes is False
    assert trips[0].reason == "dirty_vessel_skipped"


def test_roundtrip_falls_back_to_image_index_when_no_center():
    loaded = make_loaded_patient(n_slices=8)
    roi, _ = stamp_calcium_square(loaded, slice_idx=2, hu=300)
    roi_no_center = replace(roi, center_xyz=None)
    pr = make_parse_result("TEST", image_index=2, roi=roi_no_center)

    trips = xml_roundtrip_check(pr, _ct_array(loaded), loaded.slice_positions)
    t = trips[0]
    assert t.matched_via == "image_index_fallback"
    assert t.matched_slice_idx == 2
    assert t.passes is True


def test_roundtrip_rejects_center_outside_scan_even_with_valid_image_index():
    """An explicit Center that matches no slice must
    fail, not pass through the ImageIndex fallback on the stamped slice."""
    loaded = make_loaded_patient(n_slices=8)
    roi, _ = stamp_calcium_square(loaded, slice_idx=2, hu=300)
    outside = (roi.center_xyz[0], roi.center_xyz[1], max(loaded.slice_positions) + 50.0)
    pr = make_parse_result("TEST", image_index=2, roi=replace(roi, center_xyz=outside))

    t = xml_roundtrip_check(pr, _ct_array(loaded), loaded.slice_positions)[0]
    assert t.passes is False
    assert t.matched_via == "center_mismatch"
    assert t.matched_slice_idx is None
    assert t.reason == "center_outside_scan"
    assert failed_roi_ids([t]) == {(2, 0)}


def test_pass_rate_excludes_dirty():
    loaded = make_loaded_patient()
    roi_good, _ = stamp_calcium_square(loaded, slice_idx=2, hu=300)
    pr = make_parse_result("TEST", image_index=2, roi=roi_good)
    trips_ok = xml_roundtrip_check(pr, _ct_array(loaded), loaded.slice_positions)

    dirty = replace(roi_good, vessel=None, vessel_raw="X")
    pr2 = make_parse_result("TEST", image_index=2, roi=dirty)
    trips_dirty = xml_roundtrip_check(pr2, _ct_array(loaded), loaded.slice_positions)

    assert pass_rate(trips_ok) == 1.0
    assert pass_rate(trips_dirty) == 1.0  # only ROI is dirty -> excluded -> 1.0


def test_roi_idx_in_slice_recorded():
    loaded = make_loaded_patient()
    roi, _ = stamp_calcium_square(loaded, slice_idx=2)
    pr = make_parse_result("TEST", image_index=2, roi=roi)
    trips = xml_roundtrip_check(pr, _ct_array(loaded), loaded.slice_positions)
    assert trips[0].roi_idx_in_slice == 0


def test_failed_roi_ids_returns_only_failing_non_dirty():
    loaded = make_loaded_patient(n_slices=10)
    roi_ok, _ = stamp_calcium_square(loaded, slice_idx=2, hu=300)
    pr_ok = make_parse_result("TEST", image_index=2, roi=roi_ok)
    trips_ok = xml_roundtrip_check(pr_ok, _ct_array(loaded), loaded.slice_positions)
    assert failed_roi_ids(trips_ok) == set()

    # Force a failure by pointing center at wrong slice.
    wrong_center = (roi_ok.center_xyz[0], roi_ok.center_xyz[1], loaded.slice_positions[7])
    bad_roi = replace(roi_ok, center_xyz=wrong_center)
    pr_bad = make_parse_result("TEST", image_index=2, roi=bad_roi)
    trips_bad = xml_roundtrip_check(pr_bad, _ct_array(loaded), loaded.slice_positions)
    assert failed_roi_ids(trips_bad) == {(2, 0)}

    # Dirty ROIs are not in failed_roi_ids even though passes=False.
    dirty = replace(roi_ok, vessel=None, vessel_raw="X")
    pr_dirty = make_parse_result("TEST", image_index=2, roi=dirty)
    trips_dirty = xml_roundtrip_check(pr_dirty, _ct_array(loaded), loaded.slice_positions)
    assert failed_roi_ids(trips_dirty) == set()


def test_trips_to_rows_is_dictlike():
    loaded = make_loaded_patient()
    roi, _ = stamp_calcium_square(loaded, slice_idx=1)
    pr = make_parse_result("TEST", image_index=1, roi=roi)
    rows = trips_to_rows(xml_roundtrip_check(pr, _ct_array(loaded), loaded.slice_positions))
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    assert "delta_mean" in rows[0]
    assert "matched_via" in rows[0]
