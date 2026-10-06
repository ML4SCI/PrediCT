# D044 Stage 10: figure text follows the files and the decisions; one published figure set

**Date**: 2026-10-05
**Stage**: figures (stage 10)
**Status**: Active
**Module**: `scripts/10_figures.py`, `src/predict/figures/*.py`, `src/predict/stage9.py`

## Decision

1. **No claim beyond the evidence on a figure.** Figure 1 (now `figure_1_patient_pca`) and Figure 2 make no continuum claim; Figure 2's caption carries the D021.1 reading (gap results are descriptive). Figure 4 (now `figure_4_burden_composition`) names its lesion classes by their median-threshold rule (not "GMM k=3") and describes composition across burden tertiles as cross-sectional (no "maturation trajectory"). Figure 5 reports the superior-inferior coordinate within each carrier's RCA lesions (0 = lowest lesion, +z superior) and makes no proximal / distal claim; it reads stage 9's own per-carrier comparison and C8 statistics and fails when no cluster matches the C8 signature. Figure 3 reads its Hennig values and D024 interpretation from the files.
2. **One published set.** Stage 10 writes `outputs/figures.partial/` and publishes it with `COMPLETE_stage10.json` (stage 6, 7 and 9 record hashes, code fingerprint, output hashes) only when every figure built; a failure exits 1 and publishes nothing. `--only` / `--skip` runs write to `outputs/figures_smoke/` (PARTIAL_SMOKE_DO_NOT_USE).
3. **Stage 9 fingerprint scope.** Stage 9 hashes `figures/__init__.py` and `figures/_lesion_classes.py` (the only figures module the chain imports), not the whole figures package, so figure edits no longer invalidate stage 9 (amends D043's fingerprint scope).

## Rationale

Figure 5 labelled `(z - min z) / range` as "0 = most proximal" and the C8 lesions "distal"; with +z superior the label was inverted (the C8 lesions sit higher, relative z about 0.70 vs 0.14), and height alone cannot establish position along the RCA. Its comparator pooled RCA lesions of every patient while stage 9 compares within carriers. Figure 2 called the gap curves "the signature of a continuum" against D021.1; Figure 1 was titled "Burden continuum". Figure 4 mislabelled threshold groups as GMM classes (ARI with the GMM k=3 partition 0.207) and implied a time course. Figure 3 printed fixed Hennig values and a fixed verdict. Figures were saved straight into the published folder with no record. The lesion-morphology findings and the stage 9 C8 description used the same proximal / distal wording and were corrected to the superior-inferior wording.

## Consequence

Narrowing the stage 9 fingerprint invalidates an existing stage 9 record once (stages 9 and 10 are recomputed); stages 1 to 8 are unaffected.

## Verified by

- `tests/test_figures_stage10.py`: a complete run publishes with a record, a failing figure keeps the previous set, `--only` goes to the smoke folder; Figures 1 and 2 carry no continuum claim and Figure 2 cites D021.1; Figure 5 uses superior-inferior wording and stage 9's per-carrier medians, and refuses a missing or mismatched C8 cluster; Figure 4 names classes by rule with no progression wording.
