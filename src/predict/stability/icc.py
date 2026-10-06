"""ICC(3,1) absolute-agreement implementation for stage-4 stability gating,
plus ICC(1,1) (:func:`icc_1_1`) for the D033 random-draw gate.

Mathematical reference: Shrout and Fleiss 1979 / McGraw and Wong 1996,
two-way mixed-effects, single measurement, absolute agreement.

Public API:

* :func:`icc_3_1_absolute` takes a (n_subjects, k_raters) matrix and returns
  the scalar ICC. Handles NaN rows by listwise deletion. Returns NaN if the
  matrix is degenerate (fewer than 2 complete subjects, or zero total variance).
* :func:`gate_features` consumes a per-feature ICC dict, applies the threshold
  from config (D013), and returns the list of features that pass.
* :func:`build_reliability_matrix` reshapes per-perturbation feature CSVs into
  an (n_subjects, k_raters) matrix for one feature, used by the stage-5 script.

Numerical conventions:

* The formula:

      ICC(3,1) = (MSR - MSE) /
                 (MSR + (k - 1) * MSE + (k / n) * (MSC - MSE))

  where MSR is the between-subjects mean square, MSC is the between-raters
  mean square, MSE is the residual mean square, n is the number of subjects,
  and k is the number of raters.
* Negative ICCs are returned as-is (clipped only at the gate step, since a
  negative ICC means worse-than-random agreement and should fail the threshold
  by simple arithmetic).
* Zero-variance columns (a feature that returns the same constant on every
  subject regardless of perturbation) produce an undefined ICC; we return NaN
  and downstream code interprets NaN as a gate failure. The icc_source
  bypass (D016) is the correct path for genuinely invariant features.

Decisions referencing this module:
    D013 - ICC formulation and threshold.
    D016 - Geometric bypass via icc_source tagging (consumed by gate_features).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


IccSource = Literal["empirical", "invariant_by_construction"]


# ───────────────────────── core formula ─────────────────────────


def icc_3_1_absolute(matrix: np.ndarray) -> float:
    """Compute ICC(3,1) absolute agreement on an (n_subjects, k_raters) matrix.

    Rows containing any NaN are dropped (listwise deletion) before computation.
    Returns NaN if fewer than 2 complete rows remain or if the total sum of
    squares is zero.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {arr.shape}")

    # Listwise deletion.
    complete = arr[~np.isnan(arr).any(axis=1)]
    n, k = complete.shape
    if n < 2 or k < 2:
        return float("nan")

    grand_mean = complete.mean()
    subject_means = complete.mean(axis=1)
    rater_means = complete.mean(axis=0)

    ss_subject = k * float(((subject_means - grand_mean) ** 2).sum())
    ss_rater = n * float(((rater_means - grand_mean) ** 2).sum())
    ss_total = float(((complete - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_subject - ss_rater

    if ss_total == 0.0:
        # All values identical across both axes => degenerate; return NaN.
        return float("nan")

    ms_subject = ss_subject / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = (
        ms_subject
        + (k - 1) * ms_error
        + (k * (ms_rater - ms_error) / n)
    )
    if denom == 0.0:
        return float("nan")
    return float((ms_subject - ms_error) / denom)


def icc_1_1(matrix: np.ndarray, *, alpha: float = 0.05) -> tuple[float, float, float]:
    """ICC(1,1), one-way random effects, single measurement (D033).

    For designs where each subject's k measurements are random draws, not
    the same raters across subjects (Shrout and Fleiss 1979, case 1):

        ICC(1,1) = (MSB - MSW) / (MSB + (k - 1) * MSW)

    with MSB the between-subject and MSW the within-subject mean square.
    Returns ``(icc, ci_low, ci_high)``; the (1 - alpha) CI uses the F
    distribution with df (n - 1, n(k - 1)) (McGraw and Wong 1996).

    Rows with any NaN are dropped (listwise deletion; D033 counts and caps
    these separately). NaN triple if fewer than 2 rows or k < 2, or if MSW
    and MSB are both zero. If only MSW is zero the measurements agree
    perfectly: ICC 1.0.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {arr.shape}")
    complete = arr[~np.isnan(arr).any(axis=1)]
    n, k = complete.shape
    nan3 = (float("nan"), float("nan"), float("nan"))
    if n < 2 or k < 2:
        return nan3

    subject_means = complete.mean(axis=1)
    grand_mean = complete.mean()
    ms_between = k * float(((subject_means - grand_mean) ** 2).sum()) / (n - 1)
    ms_within = float(((complete - subject_means[:, None]) ** 2).sum()) / (n * (k - 1))

    if ms_within == 0.0:
        return nan3 if ms_between == 0.0 else (1.0, 1.0, 1.0)

    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)

    from scipy.stats import f as f_dist
    df1, df2 = n - 1, n * (k - 1)
    f_obs = ms_between / ms_within
    f_low = f_obs / f_dist.ppf(1.0 - alpha / 2.0, df1, df2)
    f_high = f_obs * f_dist.ppf(1.0 - alpha / 2.0, df2, df1)
    ci_low = (f_low - 1.0) / (f_low + k - 1.0)
    ci_high = (f_high - 1.0) / (f_high + k - 1.0)
    return float(icc), float(ci_low), float(ci_high)


# ───────────────────────── matrix assembly ─────────────────────────


def build_reliability_matrix(
    feature_name: str,
    baseline_df: pd.DataFrame,
    perturbation_dfs: dict[str, pd.DataFrame],
    *,
    pid_col: str = "pid",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Stack one feature's column from baseline + each perturbation CSV.

    Parameters
    ----------
    feature_name
        Column name to extract from every input frame.
    baseline_df
        Stage-3 features dataframe. Must contain ``pid_col`` and ``feature_name``.
    perturbation_dfs
        Mapping ``{perturbation_name: dataframe}``. Each dataframe is
        expected to have one row per patient with the same ``pid_col``.
    pid_col
        Patient ID column name.

    Returns
    -------
    matrix
        ``(n_subjects, k_raters)`` float array. Patients are intersected
        across all frames; perturbation order is the dict iteration order
        with baseline always first.
    pids
        Patient IDs in row order.
    raters
        Rater (perturbation) names in column order, baseline first.
    """
    if pid_col not in baseline_df.columns:
        raise KeyError(f"baseline_df missing {pid_col!r} column")
    if feature_name not in baseline_df.columns:
        raise KeyError(f"baseline_df missing feature {feature_name!r}")

    # Intersect patients across baseline + all perturbations.
    pid_sets = [set(baseline_df[pid_col].astype(str))]
    for name, df in perturbation_dfs.items():
        if pid_col not in df.columns:
            raise KeyError(f"perturbation {name!r} missing {pid_col!r}")
        if feature_name not in df.columns:
            raise KeyError(f"perturbation {name!r} missing feature {feature_name!r}")
        pid_sets.append(set(df[pid_col].astype(str)))

    common = sorted(set.intersection(*pid_sets), key=lambda s: int(s) if s.isdigit() else s)
    if not common:
        return np.empty((0, 0), dtype=float), [], []

    rater_names = ["baseline"] + list(perturbation_dfs.keys())
    cols: list[np.ndarray] = []

    def _series(df: pd.DataFrame) -> np.ndarray:
        idx = df.set_index(df[pid_col].astype(str))[feature_name]
        return idx.reindex(common).to_numpy(dtype=float)

    cols.append(_series(baseline_df))
    for name in perturbation_dfs:
        cols.append(_series(perturbation_dfs[name]))

    matrix = np.column_stack(cols)
    return matrix, common, rater_names


# ───────────────────────── gating ─────────────────────────


@dataclass(frozen=True)
class IccRecord:
    """One row of the ICC report (per feature)."""
    feature: str
    icc: float
    icc_source: IccSource
    n_subjects: int       # complete rows used in computation; 0 for bypass
    n_raters: int         # number of perturbations + baseline; 0 for bypass
    passes_gate: bool


def gate_features(
    records: list[IccRecord],
    *,
    threshold: float,
) -> tuple[list[IccRecord], list[str]]:
    """Apply the ICC threshold (D013) to a list of records.

    `passes_gate` is recomputed from `icc >= threshold` to keep the rule
    authoritative; the input value is overwritten so callers cannot disagree
    with the threshold accidentally. NaN ICC always fails.

    Returns the updated records and the list of feature names that pass.
    """
    updated: list[IccRecord] = []
    passing: list[str] = []
    for r in records:
        passes = (not np.isnan(r.icc)) and (r.icc >= threshold)
        updated.append(IccRecord(
            feature=r.feature,
            icc=r.icc,
            icc_source=r.icc_source,
            n_subjects=r.n_subjects,
            n_raters=r.n_raters,
            passes_gate=bool(passes),
        ))
        if passes:
            passing.append(r.feature)
    return updated, passing


# ───────────────────────── icc_source registry ─────────────────────────


def invariant_by_construction_features() -> tuple[str, ...]:
    """All 68 canonical features bypass the empirical gate (D016).

    Tagged ICC = 1.0 by construction. The bypass is justified architecturally:
    every canonical feature reads from ``parse_result`` (the XML's frozen
    Max / Mean fields and polygon vertices) or from ``Lesion`` objects built
    from those polygons. None of them index into the CT array. The
    perturbations in D014 transform the CT only and leave the XML unchanged,
    so canonical-feature values are bit-identical across all 14 perturbations.

    Verified by:
    - Code inspection: greping ``src/predict/features/`` for any of
      ``ct_array``, ``ct_np``, ``ct_sitk``, ``sitk.GetArrayFromImage``,
      ``_ct.npy``, ``np.load`` returns zero hits across agatston.py,
      density_tiers.py, per_vessel_aggregates.py, lesion_ccl.py,
      slice_matcher.py, spatial.py, feature_schema.py.
    - Empirical: a stage-3 worker run twice on the same patient produces
      byte-identical values for all 68 canonical columns regardless of
      external state.
    - Architectural: ``roi.max_hu`` and ``roi.mean_hu`` are set at XML parse
      time (xml_parser.py lines 122-123) from the OsiriX-stored Max / Mean
      and never mutated afterward. They are the radiologist's reading, not
      a CT-derived statistic.

    Listed via ``feature_names()`` (the schema's single source of truth) so
    this registry can never drift from the schema.
    """
    from predict.features.feature_schema import feature_names
    return feature_names()


__all__ = [
    "IccRecord",
    "IccSource",
    "icc_3_1_absolute",
    "icc_1_1",
    "build_reliability_matrix",
    "gate_features",
    "invariant_by_construction_features",
]
