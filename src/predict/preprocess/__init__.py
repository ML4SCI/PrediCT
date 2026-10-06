"""Stage 2: preprocess.

Slice matching, mask building, resampling, HU handling.
"""
from predict.preprocess.hu_handling import clip_hu, flag_metal_artifact
from predict.preprocess.mask_builder import (
    MaskBuildReport,
    build_3d_mask,
    mask_to_sitk,
)
from predict.preprocess.resampling import resample_to_target
from predict.preprocess.slice_matcher import (
    fallback_image_index_to_slice,
    match_roi_to_slice,
    resolve_roi_slice,
)

__all__ = [
    "MaskBuildReport",
    "build_3d_mask",
    "clip_hu",
    "fallback_image_index_to_slice",
    "flag_metal_artifact",
    "mask_to_sitk",
    "match_roi_to_slice",
    "resample_to_target",
    "resolve_roi_slice",
]
