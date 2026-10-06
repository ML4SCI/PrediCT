"""Tests for predict.analyse.paper_table (D028).

Coverage:
* _category_percentages on known input
* _agatston_tertile_labels produces equal-sized tertiles on smooth data
* build_paper_table produces 15 rows (3 cohorts x 5 clusters)
* required columns are present
* hennig_lookup populates the hennig_jaccard_median column
* build_robust_sensitivity_table produces 5 rows and excludes
  low_burden_flag=True patients
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predict.analyse.paper_table import (
    _agatston_tertile_labels,
    _category_percentages,
    build_paper_table,
    build_robust_sensitivity_table,
)


# ─────────────────────── helpers ───────────────────────


def _make_cohort_inputs(
    cohort: str,
    n: int = 60,
    kernel_split: tuple[int, int] = (30, 30),
    low_burden_fraction: float = 0.30,
    seed: int = 0,
) -> dict:
    """Build a synthetic per-cohort input bundle."""
    rng = np.random.default_rng(seed)
    pids = [f"{cohort}_p{i:03d}" for i in range(n)]
    agatston = rng.uniform(0, 1000, n)
    kernels = (
        ["Qr36d/2"] * kernel_split[0]
        + ["I30f/3"] * kernel_split[1]
    )
    n_low_burden = int(n * low_burden_fraction)
    low_burden_flags = (
        [True] * n_low_burden + [False] * (n - n_low_burden)
    )
    rng.shuffle(low_burden_flags)
    rng.shuffle(kernels)
    cohort_meta = pd.DataFrame({
        "pid": pids,
        "agatston_total": agatston,
        "kernel": kernels,
        "low_burden_flag": low_burden_flags,
        "category": ["cat_x"] * n,
    })
    # Spatial labels: half focal, half diffuse
    spatial_labels = pd.Series(
        ["focal"] * (n // 2) + ["diffuse"] * (n - n // 2),
        index=pids,
    )
    # Signatures: 3 features each for focal and diffuse
    sig_rows = []
    for cluster in ("focal", "diffuse"):
        for i in range(3):
            sig_rows.append({
                "cohort": cohort,
                "partition": "spatial_k2",
                "cluster": cluster,
                "rank": i + 1,
                "feature": f"sig_feat_{i}",
                "cliffs_delta": 0.5 - i * 0.1,
                "abs_cliffs_delta": 0.5 - i * 0.1,
                "direction": "up",
                "fdr_bh_pval": 0.001,
                "is_robust_discriminator": True,
            })
    signatures_df = pd.DataFrame(sig_rows)
    return {
        "cohort_metadata": cohort_meta,
        "spatial_labels": spatial_labels,
        "signatures": signatures_df,
    }


# ─────────────────────── _category_percentages ───────────────────────


def test_category_percentages_known_distribution():
    agatston = pd.Series([0.0, 50.0, 200.0, 500.0])
    pcts = _category_percentages(agatston)
    assert pcts["pct_zero"] == 25.0
    assert pcts["pct_cat_1_99"] == 25.0
    assert pcts["pct_cat_100_399"] == 25.0
    assert pcts["pct_cat_ge_400"] == 25.0


def test_category_percentages_all_zero():
    agatston = pd.Series([0.0, 0.0, 0.0])
    pcts = _category_percentages(agatston)
    assert pcts["pct_zero"] == 100.0
    assert pcts["pct_cat_1_99"] == 0.0


def test_category_percentages_fractional_boundaries():
    """Fractional scores below 1, between 99 and 100 and
    between 399 and 400 fell into no category. Every score now lands in
    exactly one, and the percentages sum to 100."""
    agatston = pd.Series([0.0, 0.4, 1.0, 99.0, 99.6, 100.0, 399.5, 400.0])
    pcts = _category_percentages(agatston)
    assert pcts["pct_zero"] == 12.5                  # 0
    assert pcts["pct_cat_1_99"] == 50.0              # 0.4, 1, 99, 99.6
    assert pcts["pct_cat_100_399"] == 25.0           # 100, 399.5
    assert pcts["pct_cat_ge_400"] == 12.5            # 400
    assert sum(pcts.values()) == pytest.approx(100.0)


def test_category_percentages_negative_score_raises():
    with pytest.raises(ValueError, match="negative"):
        _category_percentages(pd.Series([-1.0, 10.0]))


def test_category_percentages_empty_returns_nan():
    agatston = pd.Series([], dtype=float)
    pcts = _category_percentages(agatston)
    for v in pcts.values():
        assert np.isnan(v)


# ─────────────────────── _agatston_tertile_labels ───────────────────────


def test_agatston_tertile_labels_equal_sized():
    pids = [f"p{i:02d}" for i in range(30)]
    agatston = pd.Series(np.arange(30, dtype=float), index=pids)
    tertiles = _agatston_tertile_labels(agatston)
    counts = tertiles.value_counts().to_dict()
    assert counts["low"] == 10
    assert counts["mid"] == 10
    assert counts["high"] == 10


def test_agatston_tertile_labels_propagate_nan():
    agatston = pd.Series([np.nan, 10.0, 20.0, np.nan, 30.0, 40.0, 50.0])
    tertiles = _agatston_tertile_labels(agatston)
    assert pd.isna(tertiles.iloc[0])
    assert pd.isna(tertiles.iloc[3])


# ─────────────────────── build_paper_table ───────────────────────


def test_build_paper_table_produces_15_rows():
    inputs = {
        "full": _make_cohort_inputs("full", seed=0),
        "Qr36d/2": _make_cohort_inputs("qr", seed=1, kernel_split=(60, 0)),
        "I30f/3": _make_cohort_inputs("i30", seed=2, kernel_split=(0, 60)),
    }
    table = build_paper_table(inputs)
    assert len(table) == 15
    # 3 cohorts x 5 clusters
    assert set(table["cohort"]) == {"full", "Qr36d/2", "I30f/3"}
    # Each cohort has 2 spatial + 3 burden = 5 rows
    for cohort in inputs:
        cohort_rows = table[table["cohort"] == cohort]
        assert len(cohort_rows) == 5
        partitions = set(cohort_rows["partition"])
        assert partitions == {"spatial_k2", "burden_k3"}


def test_build_paper_table_required_columns_present():
    inputs = {"full": _make_cohort_inputs("full")}
    table = build_paper_table(inputs)
    required = {
        "cohort", "partition", "cluster", "N",
        "agatston_median", "agatston_iqr_lower", "agatston_iqr_upper",
        "pct_qr36d_2", "pct_i30f_3", "pct_low_burden",
        "pct_zero", "pct_cat_1_99", "pct_cat_100_399", "pct_cat_ge_400",
        "top_signature_features", "hennig_jaccard_median",
    }
    assert required <= set(table.columns)


def test_build_paper_table_N_column_is_integer():
    inputs = {"full": _make_cohort_inputs("full")}
    table = build_paper_table(inputs)
    for n in table["N"]:
        assert isinstance(n, (int, np.integer))


def test_build_paper_table_signature_text_populated_for_spatial():
    inputs = {"full": _make_cohort_inputs("full")}
    table = build_paper_table(inputs)
    spatial_rows = table[table["partition"] == "spatial_k2"]
    for _, row in spatial_rows.iterrows():
        # Signatures were provided for spatial_k2, so text should be non-empty
        assert "sig_feat" in row["top_signature_features"]


def test_build_paper_table_signature_text_empty_for_burden_k3():
    """burden_k3 partition has no signatures in the test inputs, so
    text should be empty."""
    inputs = {"full": _make_cohort_inputs("full")}
    table = build_paper_table(inputs)
    burden_rows = table[table["partition"] == "burden_k3"]
    for _, row in burden_rows.iterrows():
        assert row["top_signature_features"] == ""


def test_build_paper_table_hennig_lookup_populates_column():
    inputs = {"full": _make_cohort_inputs("full")}
    hennig_lookup = {
        ("full", "spatial_k2", "focal"): 0.88,
        ("full", "spatial_k2", "diffuse"): 0.87,
    }
    table = build_paper_table(inputs, hennig_lookup=hennig_lookup)
    spatial_focal = table[
        (table["partition"] == "spatial_k2")
        & (table["cluster"] == "focal")
    ].iloc[0]
    assert spatial_focal["hennig_jaccard_median"] == 0.88
    # Rows not in lookup get NaN
    burden_low = table[
        (table["partition"] == "burden_k3")
        & (table["cluster"] == "low")
    ].iloc[0]
    assert np.isnan(burden_low["hennig_jaccard_median"])


def test_build_paper_table_burden_tertiles_balanced():
    inputs = {"full": _make_cohort_inputs("full", n=60)}
    table = build_paper_table(inputs)
    burden_rows = table[table["partition"] == "burden_k3"]
    # 60 patients / 3 tertiles = 20 per tertile
    for n in burden_rows["N"]:
        assert 19 <= n <= 21


def test_build_paper_table_kernel_percentages_sum_correctly():
    inputs = {"full": _make_cohort_inputs("full", kernel_split=(30, 30))}
    table = build_paper_table(inputs)
    for _, row in table.iterrows():
        if row["N"] > 0:
            # pct_qr36d_2 + pct_i30f_3 should sum to ~100 (no other kernels)
            total = row["pct_qr36d_2"] + row["pct_i30f_3"]
            assert abs(total - 100.0) < 0.01


# ─────────────────────── build_robust_sensitivity_table ───────────────────────


def test_robust_sensitivity_table_5_rows():
    full_inputs = _make_cohort_inputs("full", n=60, low_burden_fraction=0.30)
    table = build_robust_sensitivity_table(full_inputs)
    assert len(table) == 5
    assert set(table["partition"]) == {"spatial_k2", "burden_k3"}


def test_robust_sensitivity_table_excludes_low_burden_patients():
    full_inputs = _make_cohort_inputs("full", n=60, low_burden_fraction=0.30)
    # Total robust patients should be 60 * (1 - 0.30) = 42
    table = build_robust_sensitivity_table(full_inputs)
    # spatial_k2 total = focal + diffuse = whole cohort size
    spatial_total = table[table["partition"] == "spatial_k2"]["N"].sum()
    assert spatial_total <= 42  # Could be less if a focal/diffuse cluster
                                 # had no remaining patients post-filter


def test_robust_sensitivity_table_cohort_label():
    full_inputs = _make_cohort_inputs("full")
    table = build_robust_sensitivity_table(
        full_inputs, robust_cohort_label="robust_280",
    )
    assert (table["cohort"] == "robust_280").all()


def test_robust_sensitivity_table_recomputes_burden_tertiles():
    """Tertiles within the robust subset should sum to the robust N,
    not to the full cohort N."""
    full_inputs = _make_cohort_inputs("full", n=60, low_burden_fraction=0.50)
    table = build_robust_sensitivity_table(full_inputs)
    burden_total = table[table["partition"] == "burden_k3"]["N"].sum()
    # Robust subset is ~30 patients; tertiles should sum to ~30
    assert burden_total <= 30


def test_robust_sensitivity_table_uses_recomputed_signatures():
    """With the full cohort's signatures (cohort 'full')
    the 'robust' rows had blank signature text."""
    full_inputs = _make_cohort_inputs("full")
    blank = build_robust_sensitivity_table(full_inputs)
    assert (blank["top_signature_features"] == "").all()
    robust_sigs = full_inputs["signatures"].assign(cohort="robust")
    table = build_robust_sensitivity_table(full_inputs, robust_signatures=robust_sigs)
    spatial = table[table["partition"] == "spatial_k2"]
    assert spatial["top_signature_features"].str.contains("sig_feat").all()


def test_robust_sensitivity_table_with_hennig_lookup():
    full_inputs = _make_cohort_inputs("full")
    hennig_lookup = {
        ("robust", "spatial_k2", "focal"): 0.91,
    }
    table = build_robust_sensitivity_table(full_inputs, hennig_lookup)
    focal_row = table[
        (table["partition"] == "spatial_k2")
        & (table["cluster"] == "focal")
    ].iloc[0]
    assert focal_row["hennig_jaccard_median"] == 0.91
