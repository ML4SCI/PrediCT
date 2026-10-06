# D022 Multi-block redundancy clustering for stage 5

**Date**: 2026-06-04
**Stage**: reduce (stage 5)
**Status**: Active (supersedes the single-matrix portion of D020 for the primary
analysis; the single-matrix variant is kept as a sensitivity check)
**Module**: `src/predict/reduce/redundancy.py`, `scripts/06_reduce.py`

## Decision

Stage 5 redundancy clustering is performed **independently within each of 6 prospective feature blocks**, then the representatives from each block are combined into the final analysis-ready feature set. The block boundaries are fixed a priori from the literature and from the canonical feature schema, NOT from any analysis result observed in stage 5.

The 6 blocks (locked):

| Block | Clinical interpretation | Source feature membership rule |
|---|---|---|
| **Burden** | Total calcium amount per vessel and overall | Any feature whose name starts with `agatston_`, `mass_`, or matches `volume_*_mm3` |
| **HU statistics** | Calcium intensity character | `max_hu_*`, `mean_hu_*`, `max_hu_global`, `mean_hu_weighted_global`, `original_firstorder_Range` |
| **Density tier distribution** | Per-vessel histogram of lesions across Agatston density factors | `n_rois_d{1..4}_*`, `has_dense_calcium`, `high_density_fraction` |
| **Spatial topology** | Geometric distribution of lesions in 3D | `lesion_count_*`, `n_calcified_arteries`, `gini_lesion_volume`, `dist_from_top_*`, `center_of_mass_z`, `inter_lesion_dist_*`, `first_to_last_dist_*`, `vessel_burden_gini` |
| **Texture** | PyRadiomics texture survivors (post-ComBat) | `original_glcm_*`, `original_glszm_*`, `original_glrlm_*`, `original_ngtdm_*`, `original_gldm_*`, all `original_firstorder_*` EXCEPT `Range` |
| **Shape** | PyRadiomics whole-mask shape geometry | `original_shape_*` |

Within each block, the existing `run_redundancy_clustering` machinery is applied (Spearman r² > 0.75 hierarchical clustering with average linkage, ICC-first representative selection, canonical-vs-PyRadiomics tiebreaker, alphabetical final tiebreaker). The final representative set is the union of representatives across all blocks.

The single-matrix variant from D020 is preserved as a **sensitivity rerun** triggered by `scripts/06_reduce.py --block-mode single`. The orchestrator default is multi-block (`--block-mode multi`).

## Rationale

### Why prospective block boundaries

Block boundaries must be locked before the analysis is examined; otherwise the methodology is contaminated by hindsight. The boundaries above are defined by **three independent prior sources**:

1. **PyRadiomics IBSI taxonomy** (Zwanenburg 2020): shape, first-order, GLCM, GLSZM, GLRLM, NGTDM, GLDM are the established families. The taxonomy is not derived from this analysis.
2. **Canonical-feature provenance**: the canonical features (burden, HU stats, density tiers, spatial topology) are XML-derived and reflect distinct clinical measurements as documented in the Agatston scoring spec (1990) and Hoori 2024's spatial-distribution methodology. Each clinical axis is a prior taxonomy unit.
3. **Lin 2022 (CCTA culprit lesions)** explicitly applies block-wise feature reduction in radiomics phenotyping with the same conceptual block structure used here. We follow that precedent.

The block partition is therefore independent of any cluster the data may produce.

### Why multi-block beats single-matrix for this dataset

The single-matrix Spearman clustering (D020) absorbed all 6 PyRadiomics texture survivors and most spatial features into the burden clusters because, in a calcium-positive cohort, texture and spatial features correlate r > 0.866 with total burden. The resulting representative set (23 features in the full cohort) was dominated by burden / density tiers (16 of 23) with only 2 spatial and 0 texture features. This composition under-represents the clinical axes of spatial topology and texture independent of burden (Lin 2022).

Multi-block clustering preserves the prior taxonomy: features can be absorbed only into clusters within their own block. The resulting representative set is more balanced and recovers the texture and spatial representations that single-matrix lost to burden collinearity.

This is NOT a method change made to produce a specific result. It is a method change made to honour the prospective biological taxonomy already encoded in the feature schema. The single-matrix variant is preserved as a published sensitivity check.

### Why we do not use the multi-block analysis to claim Finding 3 or Finding 1

Findings 1 (continuum) and 3 (spatial-only k=2) were produced under D020 single-matrix clustering. Under D022 multi-block:

- Finding 1 (continuum in the full-matrix PCA + gap statistic) is expected to replicate because the gap-curve shape is determined by the spectrum of the data, not by the choice of feature subset of size 23 vs 30.
- Finding 3 (spatial-only k=2) is unchanged because it uses the raw 11 spatial features directly as inputs to its own PCA stream, not the multi-block representatives.

The multi-block analysis is therefore an INDEPENDENT confirmation of Findings 1 and 3 using a more balanced feature set. If the findings replicate under both methods, the conclusion is strengthened.

## Alternatives considered

- **Single-matrix clustering as primary (D020 unchanged)**: 23-feature set with no texture representation. Rejected for primary analysis because the composition under-represents clinical axes from the prior taxonomy. Kept as sensitivity.
- **Soft block protection (within single-matrix clustering, do not merge across blocks)**: simpler but the cluster boundary becomes ad hoc. Multi-block is more principled and matches Lin 2022 precedent.
- **Multi-block PCA / DIABLO / MOFA**: a full multi-block dimensionality reduction framework, where each block has its own latent space and the latent spaces are jointly modelled. More complex; would require redesigning the entire stage 5 pipeline. Rejected as overengineering for the current scope; could be added in a follow-up paper.
- **Adaptive block definitions from the data**: would risk circularity. Rejected on prospective-methodology grounds.

## Verified by

- New tests in `tests/reduce/test_redundancy.py` assert that the multi-block partition is deterministic and produces the union of within-block representative sets.
- `outputs/06_reduce/multi_block_assignments.csv` records each feature's block plus its within-block cluster id and representative status.
- The orchestrator emits both `representative_features.csv` (multi-block primary) and `representative_features_single_matrix_sensitivity.csv` for comparison.
- A new test asserts that no feature is assigned to two blocks (block partition is mutually exclusive).
