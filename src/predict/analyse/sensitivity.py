"""Stage 7 low-burden sensitivity analysis (D028).

D028 promises, on the full cohort restricted to ``low_burden_flag == False``
and with the full cohort's spatial labels kept (no reclustering): the
burden-orthogonality test (D024), the directional hypotheses (D025) and the
signatures behind the 5-row sensitivity table. The subset's signatures are
recomputed, because the full cohort's signatures carry a different cohort
label (D028.1).

Sensitivity verdict (D028.1, fixed 2026-10-05 before stage 7 ran under it): the
restricted cohort is assessed with the same rules as the full cohort
(D024 interpretation; D025 primary rule, >= 4 of 6 confirmed). Each is
reported as the same as or different from the full-cohort result. The
kernel-stratified D025 secondary rule does not apply (one cohort), and the
stage 6 Hennig values describe full-cohort clusters, so none is reported
for the restricted cohort.
"""
from __future__ import annotations

import pandas as pd

from predict.analyse.hypotheses import directional_hypotheses_table, primary_pass
from predict.analyse.orthogonality import assess_burden_orthogonality
from predict.analyse.paper_table import _agatston_tertile_labels
from predict.analyse.profiles import apply_fdr_bh, compute_cluster_profile
from predict.analyse.signatures import top_n_signatures

ROBUST_LABEL = "robust"


def robust_pids(cohort_meta: pd.DataFrame) -> list[str]:
    """Patients with ``low_burden_flag == False`` (D028)."""
    flag = cohort_meta["low_burden_flag"].astype(str).str.lower()
    unknown = sorted(set(flag) - {"true", "false", "1", "0", "1.0", "0.0"})
    if unknown:
        raise ValueError(f"unrecognised low_burden_flag values: {unknown}")
    return cohort_meta.loc[flag.isin({"false", "0", "0.0"}), "pid"].astype(str).tolist()


def low_burden_sensitivity(
    raw_features: pd.DataFrame,
    spatial_labels: pd.Series,
    cohort_meta: pd.DataFrame,
    feature_list: list[str],
    full_d024_interpretation: str,
    full_d025_primary_passes: bool,
    top_n: int = 5,
    label: str = ROBUST_LABEL,
) -> dict:
    """D024, D025 (primary rule) and signatures on the robust patients.

    ``raw_features`` is indexed by pid (raw scale, derived features added);
    ``spatial_labels`` maps pid -> "focal" / "diffuse" for the full cohort;
    ``cohort_meta`` has pid, agatston_total and low_burden_flag.
    """
    keep = pd.Index(robust_pids(cohort_meta))
    labels = spatial_labels.loc[spatial_labels.index.intersection(keep)]
    raw = raw_features.loc[raw_features.index.intersection(labels.index)]
    labels = labels.loc[raw.index]
    meta = cohort_meta.assign(pid=cohort_meta["pid"].astype(str)).set_index("pid")
    agatston = meta.loc[raw.index, "agatston_total"]

    ortho = assess_burden_orthogonality(
        agatston[labels == "focal"].to_numpy(dtype=float),
        agatston[labels == "diffuse"].to_numpy(dtype=float),
        cohort=label,
    )
    hypotheses = directional_hypotheses_table(raw, labels, cohort=label)
    primary = primary_pass(hypotheses)

    tertiles = _agatston_tertile_labels(agatston).astype(str)
    profiles = apply_fdr_bh(pd.concat([
        compute_cluster_profile(raw, labels, feature_list, cohort=label, partition="spatial_k2"),
        compute_cluster_profile(raw, tertiles, feature_list, cohort=label, partition="burden_k3"),
    ], ignore_index=True))
    signatures = top_n_signatures(profiles, n=top_n, only_robust=True)

    verdict = {
        "cohort": label,
        "n_patients": int(len(raw)),
        "n_focal": int((labels == "focal").sum()),
        "n_diffuse": int((labels == "diffuse").sum()),
        "d024_interpretation": ortho.interpretation,
        "d024_full_interpretation": full_d024_interpretation,
        "d024_same_as_full": ortho.interpretation == full_d024_interpretation,
        "d025_primary": primary,
        "d025_full_primary_passes": bool(full_d025_primary_passes),
        "d025_primary_same_as_full": bool(primary["passes"]) == bool(full_d025_primary_passes),
        "hennig": "not reported: stage 6 Hennig values describe full-cohort clusters",
    }
    return {
        "orthogonality": pd.DataFrame([ortho.to_dict()]),
        "hypotheses": hypotheses,
        "profiles": profiles,
        "signatures": signatures,
        "verdict": verdict,
    }


__all__ = ["ROBUST_LABEL", "robust_pids", "low_burden_sensitivity"]
