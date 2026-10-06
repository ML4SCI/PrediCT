"""Stage 6: cluster discovery and validity (D021).

Three modules:
  * clusterability     - Hopkins statistic with pyclustertend k=2-for-both
    convention. Threshold 0.65 for "clustered", 0.55-0.65 ambiguous band.
  * cluster_discovery  - fit_cluster (kmeans / ward / GMM), gap statistic
    (Tibshirani with Maitra-Ramler 2010 1-SE-from-argmax k-selection),
    Monti consensus, burden residualisation.
  * validity           - kernel chi-square confounder test, Hennig
    clusterboot stability (median Jaccard), ARI on shared pids for
    cross-cohort agreement.

Kept separate from ``predict.reduce`` so stage 5 stays focused on
dimensionality reduction. The script seam is
``outputs/06_reduce/{prepared_matrix,pca_scores}.csv`` -> 07_discover.py.

See ``docs/modules/discover.md`` and decision D021.
"""
from predict.discover.cluster_discovery import (
    ClusterAlgorithm,
    ConsensusResult,
    GapStatisticResult,
    burden_residualise,
    fit_cluster,
    gap_statistic,
    monti_consensus,
    within_cluster_dispersion,
)
from predict.discover.clusterability import (
    HopkinsResult,
    assess_clusterability,
    hopkins_statistic,
)
from predict.discover.validity import (
    HennigStabilityResult,
    KernelConfounderResult,
    ari,
    ari_on_shared_pids,
    hennig_clusterboot,
    kernel_chi_square,
)

__all__ = [
    # clusterability
    "HopkinsResult", "assess_clusterability", "hopkins_statistic",
    # cluster_discovery
    "ClusterAlgorithm", "ConsensusResult", "GapStatisticResult",
    "burden_residualise", "fit_cluster", "gap_statistic",
    "monti_consensus", "within_cluster_dispersion",
    # validity
    "HennigStabilityResult", "KernelConfounderResult", "ari",
    "ari_on_shared_pids", "hennig_clusterboot", "kernel_chi_square",
]
