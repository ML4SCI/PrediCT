"""Tests for predict.preprocess.mask_builder.

The critical assertion is that the rasterised polygon lands on the *same*
slice as where the calcium was stamped: i.e. the mask is not Z-flipped.
This is the regression test for a Z-flipped slice mapping (D001).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from predict.io.xml_parser import ParseResult, SliceAnnotation
from predict.preprocess.mask_builder import build_3d_mask, mask_to_sitk

from tests.preprocess._helpers import make_loaded_patient, make_parse_result, stamp_calcium_square


def test_mask_lands_on_stamped_slice():
    loaded = make_loaded_patient(n_slices=10)
    roi, _ = stamp_calcium_square(loaded, slice_idx=3)
    parse_result = make_parse_result("TEST", image_index=3, roi=roi)

    mask, report = build_3d_mask(parse_result, loaded)

    assert mask.shape[0] == 10
    assert mask[3].sum() > 0, "Mask must land on slice 3 (same as stamped calcium)"
    for k in range(10):
        if k != 3:
            assert mask[k].sum() == 0, f"Slice {k} should be empty"
    assert report.n_rasterised == 1
    assert report.n_skipped_dirty == 0
    assert report.n_skipped_no_match == 0


def test_dirty_vessel_skipped_by_default():
    loaded = make_loaded_patient()
    roi, _ = stamp_calcium_square(loaded, slice_idx=2)
    dirty_roi = replace(roi, vessel=None, vessel_raw="555614876")
    parse_result = make_parse_result("TEST", image_index=2, roi=dirty_roi)

    mask, report = build_3d_mask(parse_result, loaded)
    assert mask.sum() == 0
    assert report.n_skipped_dirty == 1
    assert report.n_rasterised == 0


def test_dirty_vessel_kept_when_flag_off():
    loaded = make_loaded_patient()
    roi, _ = stamp_calcium_square(loaded, slice_idx=2)
    dirty_roi = replace(roi, vessel=None, vessel_raw="Unnamed")
    parse_result = make_parse_result("TEST", image_index=2, roi=dirty_roi)

    mask, report = build_3d_mask(parse_result, loaded, skip_dirty=False)
    assert mask[2].sum() > 0
    assert report.n_rasterised == 1
    assert report.n_skipped_dirty == 0


def test_unmatched_roi_uses_fallback_when_no_center():
    loaded = make_loaded_patient(n_slices=8)
    roi, _ = stamp_calcium_square(loaded, slice_idx=5)
    roi_no_center = replace(roi, center_xyz=None)
    parse_result = make_parse_result("TEST", image_index=5, roi=roi_no_center)

    mask, report = build_3d_mask(parse_result, loaded)
    # Fallback maps ImageIndex 5 directly to slice 5.
    assert mask[5].sum() > 0
    assert report.n_matched_by_fallback == 1


def test_center_outside_scan_is_not_rasterised_via_fallback():
    """ImageIndex 5 is valid, but the ROI's Center
    contradicts the scan, so it must stay out of the mask."""
    loaded = make_loaded_patient(n_slices=8)
    roi, _ = stamp_calcium_square(loaded, slice_idx=5)
    outside = (roi.center_xyz[0], roi.center_xyz[1], max(loaded.slice_positions) + 50.0)
    parse_result = make_parse_result("TEST", image_index=5, roi=replace(roi, center_xyz=outside))

    mask, report = build_3d_mask(parse_result, loaded)
    assert mask.sum() == 0
    assert report.n_skipped_no_match == 1
    assert report.n_matched_by_fallback == 0


def test_mask_to_sitk_inherits_metadata():
    loaded = make_loaded_patient()
    mask = np.zeros((loaded.n_slices, 64, 64), dtype=np.uint8)
    img = mask_to_sitk(mask, loaded.ct_sitk)
    assert img.GetSpacing() == loaded.ct_sitk.GetSpacing()
    assert img.GetOrigin() == loaded.ct_sitk.GetOrigin()
    assert img.GetDirection() == loaded.ct_sitk.GetDirection()


def test_empty_parse_result_yields_empty_mask():
    loaded = make_loaded_patient()
    empty = ParseResult(pid="TEST", slices=(SliceAnnotation(image_index=0, rois=()),))
    mask, report = build_3d_mask(empty, loaded)
    assert mask.sum() == 0
    assert report.n_rois_total == 0
    assert report.n_rasterised == 0


def test_excluded_roi_ids_skip_roi():
    loaded = make_loaded_patient(n_slices=10)
    roi, _ = stamp_calcium_square(loaded, slice_idx=3)
    parse_result = make_parse_result("TEST", image_index=3, roi=roi)

    mask, report = build_3d_mask(
        parse_result, loaded,
        excluded_roi_ids={(3, 0)},  # image_index=3, roi_idx=0
    )
    assert mask.sum() == 0, "excluded ROI must not appear in mask"
    assert report.n_excluded_by_roundtrip == 1
    assert report.n_rasterised == 0


def test_excluded_roi_ids_only_skip_listed_rois():
    loaded = make_loaded_patient(n_slices=10)
    roi, _ = stamp_calcium_square(loaded, slice_idx=2)
    parse_result = make_parse_result("TEST", image_index=2, roi=roi)

    mask, report = build_3d_mask(
        parse_result, loaded,
        excluded_roi_ids={(9, 5)},  # different (image_index, roi_idx), should NOT match
    )
    assert mask[2].sum() > 0
    assert report.n_excluded_by_roundtrip == 0
    assert report.n_rasterised == 1
