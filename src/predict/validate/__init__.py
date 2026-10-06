"""Stage 8: validate.

Phenotype + correctness validation:

* :mod:`xml_roundtrip`: D002 Max-exact correctness gate (also used at stage 2).
* :mod:`label_alignment`: D023 focal/diffuse mapping for raw GMM labels (stage 8 shared helper).
"""
from predict.validate.cross_cohort_ari import (
    ARI_PASS_THRESHOLD,
    consolidate as consolidate_cross_cohort_ari,
    write as write_cross_cohort_ari,
)
from predict.validate.external_holdout import (
    FrozenPipeline,
    SPATIAL_FEATURES_FOR_PROJECTION,
    fit_frozen_pipeline,
    project_holdout_to_spatial_pca,
    run_external_holdout_validation,
    transform_holdout_features,
)
from predict.validate.leave_k_out import (
    DISAGREEMENT_RATE_DEFAULT,
    FoldSplit,
    N_SIM_DEFAULT,
    N_SPLITS_DEFAULT,
    PERCENTILE_DEFAULT,
    SEED_DEFAULT,
    attach_summary_row,
    kernel_stratified_kfold_split,
    predict_fold,
    run_leave_k_out,
    simulate_ari_threshold,
)
from predict.validate.label_alignment import (
    DIFFUSE_LABEL_INT,
    DIFFUSE_LABEL_STR,
    FOCAL_LABEL_INT,
    FOCAL_LABEL_STR,
    apply_mapping,
    canonical_numeric_labels,
    canonical_string_labels,
    determine_mapping,
)
from predict.validate.xml_roundtrip import (
    ROITrip,
    check_roi,
    failed_roi_ids,
    pass_rate,
    trips_to_rows,
    xml_roundtrip_check,
)

__all__ = [
    # xml_roundtrip
    "ROITrip",
    "check_roi",
    "failed_roi_ids",
    "pass_rate",
    "trips_to_rows",
    "xml_roundtrip_check",
    # label_alignment
    "DIFFUSE_LABEL_INT",
    "DIFFUSE_LABEL_STR",
    "FOCAL_LABEL_INT",
    "FOCAL_LABEL_STR",
    "apply_mapping",
    "canonical_numeric_labels",
    "canonical_string_labels",
    "determine_mapping",
    # cross_cohort_ari
    "ARI_PASS_THRESHOLD",
    "consolidate_cross_cohort_ari",
    "write_cross_cohort_ari",
    # external_holdout
    "FrozenPipeline",
    "SPATIAL_FEATURES_FOR_PROJECTION",
    "fit_frozen_pipeline",
    "project_holdout_to_spatial_pca",
    "run_external_holdout_validation",
    "transform_holdout_features",
    # leave_k_out
    "DISAGREEMENT_RATE_DEFAULT",
    "FoldSplit",
    "N_SIM_DEFAULT",
    "N_SPLITS_DEFAULT",
    "PERCENTILE_DEFAULT",
    "SEED_DEFAULT",
    "attach_summary_row",
    "kernel_stratified_kfold_split",
    "predict_fold",
    "run_leave_k_out",
    "simulate_ari_threshold",
]
