"""Burden-propensity matching for the LAD-phenotype experiment.

Caliper-matched 1:k nearest-neighbour matching on a single covariate
(typically log(agatston_total + 1)). Without replacement. Greedy
ordering by case rank. Post-match standardised mean difference (SMD)
diagnostic.

Reusable beyond this experiment: the same machinery applies to the
pending C8 RCA burden-independence question. Kept under
experiments/lad_phenotype/ for now; refactor to a shared module if a
third caller emerges.

Test coverage in tests/test_propensity.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatchResult:
    """Output of caliper_match. All ndarrays / Series are aligned by row.

    Attributes
    ----------
    pairs : pd.DataFrame
        Columns: case_pid, control_pid, case_value, control_value,
        abs_distance. One row per matched control. A case may appear
        in multiple rows (1:k matching).
    case_pids_matched : list[str]
        pids of cases that received at least one control. Subset of the
        cases passed in.
    case_pids_unmatched : list[str]
        pids of cases that received zero controls.
    smd_pre : float
        standardised mean difference between cases and controls BEFORE
        matching.
    smd_post : float
        standardised mean difference between the matched cases (each once)
        and the matched controls (each once): the two populations the
        post-match outcome tests compare. PASS criterion: < 0.1.
        Repeating each case once per matched control would weight the
        comparison differently from the outcome tests, so with variable
        1:1-1:3 matching the check could pass on an imbalanced comparison
        (D043).
    smd_post_pair_weighted : float
        the pair-weighted value, reported for reference.
    n_cases_in : int
    n_controls_in : int
    n_cases_matched : int
    n_controls_used : int
    caliper : float
        absolute caliper value used (typically 0.2 * sd of variable).
    """
    pairs: pd.DataFrame
    case_pids_matched: list
    case_pids_unmatched: list
    smd_pre: float
    smd_post: float
    n_cases_in: int
    n_controls_in: int
    n_cases_matched: int
    n_controls_used: int
    caliper: float
    smd_post_pair_weighted: float = float("nan")


def standardised_mean_difference(
    a: np.ndarray, b: np.ndarray,
) -> float:
    """SMD = (mean_a - mean_b) / pooled_sd. Common imbalance diagnostic.

    A value of |SMD| < 0.1 is the standard "balanced" cutoff in
    propensity-score literature (Austin 2011).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    mean_diff = float(np.mean(a) - np.mean(b))
    var_a = float(np.var(a, ddof=1)) if len(a) > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if len(b) > 1 else 0.0
    pooled = np.sqrt((var_a + var_b) / 2.0)
    if pooled == 0.0:
        return float("nan") if mean_diff != 0.0 else 0.0
    return mean_diff / pooled


def caliper_match(
    cases: pd.Series,
    controls: pd.Series,
    *,
    caliper_sd: float = 0.2,
    k: int = 3,
    random_state: int = 42,
) -> MatchResult:
    """Greedy 1:k nearest-neighbour matching on a single covariate
    with a fixed caliper.

    Parameters
    ----------
    cases : pd.Series
        Indexed by pid. Values are the matching covariate for each case.
    controls : pd.Series
        Indexed by pid. Values are the matching covariate for each
        candidate control. Must be disjoint from `cases` (no pid in
        both); the function asserts this.
    caliper_sd : float
        Caliper in units of standard deviations of the COMBINED case +
        control covariate values. Default 0.2 (Austin 2011 standard).
    k : int
        Maximum controls per case. Each control is used at most once
        across all cases (without-replacement matching).
    random_state : int
        Tie-break seed for control ordering when distances are
        identical (rare with continuous covariates).

    Returns
    -------
    MatchResult
    """
    case_pids = list(cases.index.astype(str))
    control_pids = list(controls.index.astype(str))
    overlap = set(case_pids) & set(control_pids)
    if overlap:
        raise ValueError(
            f"caliper_match: cases and controls overlap on "
            f"{len(overlap)} pids; first few: {sorted(overlap)[:5]}"
        )

    case_vals = cases.astype(float).to_numpy()
    control_vals = controls.astype(float).to_numpy()

    # Combined SD for the caliper.
    combined = np.concatenate([case_vals, control_vals])
    if combined.size <= 1 or np.std(combined, ddof=1) == 0.0:
        raise ValueError(
            "caliper_match: combined covariate has zero variance; "
            "matching is not meaningful."
        )
    combined_sd = float(np.std(combined, ddof=1))
    caliper = caliper_sd * combined_sd

    smd_pre = standardised_mean_difference(case_vals, control_vals)

    # Sort cases by their covariate rank (ascending). Greedy matching
    # in this order minimises bias from later cases stealing "easy"
    # controls; the alternative (random order) is non-reproducible.
    rng = np.random.default_rng(random_state)
    case_order = np.argsort(case_vals, kind="stable")
    # Working copies for the controls (we mutate availability).
    control_available_mask = np.ones(len(control_pids), dtype=bool)
    control_vals_arr = np.asarray(control_vals)

    pair_records: list[dict] = []
    matched_case_pids: list = []
    unmatched_case_pids: list = []

    for ci in case_order:
        c_pid = case_pids[ci]
        c_val = case_vals[ci]
        # Distances to currently available controls.
        avail_idx = np.where(control_available_mask)[0]
        if avail_idx.size == 0:
            unmatched_case_pids.append(c_pid)
            continue
        dists = np.abs(control_vals_arr[avail_idx] - c_val)
        # Within caliper.
        within = avail_idx[dists <= caliper]
        if within.size == 0:
            unmatched_case_pids.append(c_pid)
            continue
        # Take up to k smallest-distance controls, breaking ties by rng.
        within_dists = np.abs(control_vals_arr[within] - c_val)
        # Stable tie-break: add a tiny rng-jitter, sort, pick top-k.
        jitter = rng.uniform(0, 1e-12, size=within.size)
        order = np.argsort(within_dists + jitter, kind="stable")
        chosen = within[order[:k]]
        if chosen.size == 0:
            unmatched_case_pids.append(c_pid)
            continue
        matched_case_pids.append(c_pid)
        for j in chosen:
            ctrl_pid = control_pids[j]
            ctrl_val = float(control_vals_arr[j])
            pair_records.append({
                "case_pid": c_pid,
                "control_pid": ctrl_pid,
                "case_value": float(c_val),
                "control_value": ctrl_val,
                "abs_distance": float(abs(c_val - ctrl_val)),
            })
            control_available_mask[j] = False

    pairs = pd.DataFrame(pair_records)
    if pairs.empty:
        smd_post = smd_pairs = float("nan")
    else:
        # Balance of the populations actually compared: each matched case
        # once, each matched control once (controls are used at most once).
        cases_once = pairs.drop_duplicates("case_pid")["case_value"].to_numpy()
        smd_post = standardised_mean_difference(cases_once, pairs["control_value"].to_numpy())
        smd_pairs = standardised_mean_difference(
            pairs["case_value"].to_numpy(),
            pairs["control_value"].to_numpy(),
        )

    return MatchResult(
        pairs=pairs,
        case_pids_matched=matched_case_pids,
        case_pids_unmatched=unmatched_case_pids,
        smd_pre=float(smd_pre),
        smd_post=float(smd_post),
        n_cases_in=int(len(case_pids)),
        n_controls_in=int(len(control_pids)),
        n_cases_matched=int(len(matched_case_pids)),
        n_controls_used=int(len(pairs)),
        caliper=float(caliper),
        smd_post_pair_weighted=float(smd_pairs),
    )


def match_yield(result: MatchResult) -> float:
    """Fraction of cases that received at least one control."""
    if result.n_cases_in == 0:
        return float("nan")
    return result.n_cases_matched / result.n_cases_in


__all__ = [
    "MatchResult",
    "standardised_mean_difference",
    "caliper_match",
    "match_yield",
]
