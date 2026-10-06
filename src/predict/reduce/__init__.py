"""Stage 5: dimensionality reduction.

Three modules:
  * prepare_matrix - D019 preprocessing pipeline (variance filter, derived
    features, ComBat, Yeo-Johnson + rank fallback, z-score).
  * redundancy     - D020 (single-matrix) and D022 (multi-block) Spearman
    r^2 > 0.75 clustering and representative selection.
  * pca            - D020 PCA with sign normalisation on the representatives.

Cluster discovery (D021) is the discover stage; see ``predict.discover`` and
``docs/modules/discover.md``.

See ``docs/modules/reduce.md`` and decisions D017 through D020 and D022.
"""
from predict.reduce.pca import (
    PcaResult,
    assign_family,
    explained_variance_table,
    fit_pca,
    normalise_pc_signs,
    pc_external_correlation,
    select_n_retain,
    top_loadings_table,
)
from predict.reduce.prepare_matrix import (
    D017_DROPPED_FEATURES,
    D018_BINARISE_SOURCE,
    D018_BINARISE_TARGET,
    SPARSE_COLUMNS,
    harmonisable_columns,
    MatrixPrepLog,
    run_matrix_prep,
)
from predict.reduce.redundancy import (
    DEFAULT_BLOCKS,
    FeatureBlock,
    IccInfo,
    MultiBlockResult,
    RedundancyResult,
    assign_features_to_blocks,
    build_icc_lookup,
    run_multi_block_redundancy_clustering,
    run_redundancy_clustering,
)

__all__ = [
    # prepare_matrix
    "D017_DROPPED_FEATURES", "D018_BINARISE_SOURCE", "D018_BINARISE_TARGET",
    "SPARSE_COLUMNS", "MatrixPrepLog", "harmonisable_columns",
    "run_matrix_prep",
    # redundancy (single-matrix D020 + multi-block D022)
    "DEFAULT_BLOCKS", "FeatureBlock", "IccInfo", "MultiBlockResult",
    "RedundancyResult", "assign_features_to_blocks", "build_icc_lookup",
    "run_multi_block_redundancy_clustering", "run_redundancy_clustering",
    # pca
    "PcaResult", "assign_family", "explained_variance_table", "fit_pca",
    "normalise_pc_signs", "pc_external_correlation", "select_n_retain",
    "top_loadings_table",
]
