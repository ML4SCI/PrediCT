"""Tests for predict.features.per_artery_mask + the mask_builder vessel_filter."""
from __future__ import annotations

from dataclasses import replace

from predict.features.per_artery_mask import (
    build_per_artery_masks,
    per_artery_voxel_counts,
)
from predict.io.xml_parser import ROI, ParseResult, SliceAnnotation
from predict.preprocess.mask_builder import build_3d_mask

from tests.preprocess._helpers import (
    make_loaded_patient,
    make_parse_result,
    stamp_calcium_square,
)


def _two_vessel_parse_result(loaded, slice_idx_lad=3, slice_idx_rca=5):
    """Build a ParseResult with one LAD ROI and one RCA ROI on different slices."""
    roi_lad, _ = stamp_calcium_square(loaded, slice_idx=slice_idx_lad,
                                      x0=20, y0=20, side=10, hu=300)
    roi_rca, _ = stamp_calcium_square(loaded, slice_idx=slice_idx_rca,
                                      x0=40, y0=40, side=10, hu=350)
    # Re-tag RCA on the second ROI (the stamper always emits an LAD ROI).
    roi_rca = replace(
        roi_rca,
        vessel="RCA",
        vessel_raw="Right Coronary Artery",
    )
    parse_result = ParseResult(
        pid="TEST",
        slices=(
            SliceAnnotation(image_index=slice_idx_lad, rois=(roi_lad,)),
            SliceAnnotation(image_index=slice_idx_rca, rois=(roi_rca,)),
        ),
        dirty_vessel_names=(),
        n_active_rois=2,
    )
    return parse_result


# ─────────── mask_builder vessel_filter ───────────


def test_vessel_filter_isolates_one_vessel():
    loaded = make_loaded_patient(n_slices=10)
    parse = _two_vessel_parse_result(loaded)

    mask_lad, _ = build_3d_mask(parse, loaded, vessel_filter="LAD")
    mask_rca, _ = build_3d_mask(parse, loaded, vessel_filter="RCA")

    assert mask_lad[3].sum() > 0, "LAD ROI must appear on slice 3"
    assert mask_lad[5].sum() == 0, "RCA's slice must be empty when filtering LAD"
    assert mask_rca[5].sum() > 0
    assert mask_rca[3].sum() == 0


def test_vessel_filter_does_not_affect_default_call():
    """The new vessel_filter kwarg defaults to None: whole-mask unchanged."""
    loaded = make_loaded_patient(n_slices=10)
    roi, _ = stamp_calcium_square(loaded, slice_idx=4)
    parse = make_parse_result("TEST", image_index=4, roi=roi)
    mask_default, _ = build_3d_mask(parse, loaded)
    mask_explicit_none, _ = build_3d_mask(parse, loaded, vessel_filter=None)
    assert (mask_default == mask_explicit_none).all()


# ─────────── build_per_artery_masks ───────────


def test_per_artery_masks_keys_are_canonical():
    loaded = make_loaded_patient(n_slices=10)
    parse = _two_vessel_parse_result(loaded)
    masks = build_per_artery_masks(parse, loaded)
    assert set(masks.keys()) == {"LAD", "RCA", "LCx", "LM"}


def test_per_vessel_voxel_counts_sum_to_whole_mask():
    """For a parse_result with no overlap between vessels, per-vessel sum
    equals whole-mask voxel count."""
    loaded = make_loaded_patient(n_slices=10)
    parse = _two_vessel_parse_result(loaded)
    whole, _ = build_3d_mask(parse, loaded)
    masks = build_per_artery_masks(parse, loaded)
    per_v = per_artery_voxel_counts(masks)
    assert sum(per_v.values()) == int(whole.sum())


def test_empty_vessel_returns_zero_mask():
    loaded = make_loaded_patient(n_slices=10)
    parse = _two_vessel_parse_result(loaded)  # LAD + RCA only
    masks = build_per_artery_masks(parse, loaded)
    assert masks["LCx"].sum() == 0
    assert masks["LM"].sum() == 0
    # Shape is preserved even for empty vessels.
    assert masks["LCx"].shape == masks["LAD"].shape


def test_excluded_roi_ids_skipped_in_per_vessel():
    loaded = make_loaded_patient(n_slices=10)
    parse = _two_vessel_parse_result(loaded)
    # Exclude the LAD ROI on slice_idx=3 (image_index=3, roi_idx=0).
    masks = build_per_artery_masks(
        parse, loaded,
        excluded_roi_ids={(3, 0)},
    )
    assert masks["LAD"].sum() == 0, "LAD mask must be empty after exclusion"
    assert masks["RCA"].sum() > 0
