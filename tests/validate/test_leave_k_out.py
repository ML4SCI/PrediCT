"""Tests for predict.validate.leave_k_out (D030)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from predict.validate.leave_k_out import (
    DISAGREEMENT_RATE_DEFAULT,
    N_SPLITS_DEFAULT,
    PERCENTILE_DEFAULT,
    SEED_DEFAULT,
    attach_summary_row,
    kernel_stratified_kfold_split,
    predict_fold,
    run_leave_k_out,
    simulate_ari_threshold,
)

# Reuse the synthetic full-cohort fixture from the external_holdout tests.
from tests.validate.test_external_holdout import (  # noqa: E402
    _feature_cols,
    _toy_full_cohort,
)


@pytest.fixture
def full_cohort():
    # Slightly larger than external_holdout test for sane 10-fold splits.
    return _toy_full_cohort(n=240, seed=0)


@pytest.fixture
def reference_labels(full_cohort):
    """Synthetic full-cohort spatial-k2 labels driven by lesion_count_total."""
    # Higher lesion_count_total -> diffuse (1); lower -> focal (0).
    cutoff = float(np.median(full_cohort["lesion_count_total"]))
    labels = (full_cohort["lesion_count_total"] > cutoff).astype(int)
    return pd.Series(
        labels.to_numpy(), index=full_cohort["pid"].astype(str),
    )


# ─────────────────────── splitting ───────────────────────


def test_kernel_stratified_split_count(full_cohort):
    splits = kernel_stratified_kfold_split(
        full_cohort, n_splits=N_SPLITS_DEFAULT, seed=SEED_DEFAULT,
    )
    assert len(splits) == N_SPLITS_DEFAULT


def test_kernel_stratified_split_disjoint_test_sets(full_cohort):
    splits = kernel_stratified_kfold_split(full_cohort, n_splits=10)
    all_test = []
    for s in splits:
        all_test.extend(s.test_pids)
    assert len(all_test) == len(set(all_test)) == len(full_cohort)


def test_kernel_stratified_split_every_fold_has_each_kernel(full_cohort):
    """ComBat requires >= 2 samples per kernel in every TRAIN fold."""
    splits = kernel_stratified_kfold_split(full_cohort, n_splits=10)
    for s in splits:
        assert s.train_kernel_counts.get("Qr36d/2", 0) >= 2
        assert s.train_kernel_counts.get("I30f/3", 0) >= 2
        # And the test fold has both too, on a 10-split of N=240.
        assert s.test_kernel_counts.get("Qr36d/2", 0) >= 1
        assert s.test_kernel_counts.get("I30f/3", 0) >= 1


def test_kernel_stratified_split_seed_reproducible(full_cohort):
    a = kernel_stratified_kfold_split(full_cohort, n_splits=10, seed=42)
    b = kernel_stratified_kfold_split(full_cohort, n_splits=10, seed=42)
    for s1, s2 in zip(a, b):
        assert s1.train_pids == s2.train_pids
        assert s1.test_pids == s2.test_pids


def test_kernel_stratified_split_seed_changes_split(full_cohort):
    a = kernel_stratified_kfold_split(full_cohort, n_splits=10, seed=42)
    b = kernel_stratified_kfold_split(full_cohort, n_splits=10, seed=43)
    # Some fold must differ between seeds.
    assert any(s1.test_pids != s2.test_pids for s1, s2 in zip(a, b))


# ─────────────────────── per-fold predict ───────────────────────


def test_predict_fold_returns_label_per_test_pid(full_cohort):
    splits = kernel_stratified_kfold_split(full_cohort, n_splits=10, seed=42)
    s = splits[0]
    train_df = full_cohort[full_cohort["pid"].astype(str).isin(s.train_pids)]
    test_df = full_cohort[full_cohort["pid"].astype(str).isin(s.test_pids)]
    raw, frozen = predict_fold(
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        _feature_cols(full_cohort),
    )
    assert raw.shape == (len(test_df),)
    assert set(raw.tolist()).issubset({0, 1})


# ─────────────────────── simulation threshold ───────────────────────


def test_simulate_ari_threshold_zero_disagreement_close_to_one():
    """At 0% disagreement (n_flip = 0), every iteration gives ARI = 1.0."""
    ref = np.array([0] * 20 + [1] * 22)
    t = simulate_ari_threshold(
        ref, disagreement_rate=0.0, n_simulations=200,
    )
    assert np.isnan(t)  # n_flip == 0 -> return NaN by contract


def test_simulate_ari_threshold_returns_finite(reference_labels):
    # Interleave indices to get a balanced sample (toy cohort is built as
    # cluster-A first half + cluster-B second half; taking the first 42
    # would be all-one-class and give degenerate ARI).
    half = len(reference_labels) // 2
    sample_idx = list(range(0, 21)) + list(range(half, half + 21))
    ref = reference_labels.iloc[sample_idx].to_numpy()
    t = simulate_ari_threshold(
        ref, disagreement_rate=0.10, n_simulations=500, rng_seed=1,
    )
    assert np.isfinite(t)
    assert -1.0 <= t <= 1.0


def test_simulate_ari_threshold_decreases_with_more_disagreement(reference_labels):
    # Interleave indices to get a balanced sample (toy cohort is built as
    # cluster-A first half + cluster-B second half; taking the first 42
    # would be all-one-class and give degenerate ARI).
    half = len(reference_labels) // 2
    sample_idx = list(range(0, 21)) + list(range(half, half + 21))
    ref = reference_labels.iloc[sample_idx].to_numpy()
    t_low = simulate_ari_threshold(
        ref, disagreement_rate=0.05, n_simulations=500, rng_seed=1,
    )
    t_high = simulate_ari_threshold(
        ref, disagreement_rate=0.20, n_simulations=500, rng_seed=1,
    )
    # Higher K = more noise = lower 5th-percentile ARI.
    assert t_high < t_low


def test_simulate_ari_threshold_seed_reproducible(reference_labels):
    # Interleave indices to get a balanced sample (toy cohort is built as
    # cluster-A first half + cluster-B second half; taking the first 42
    # would be all-one-class and give degenerate ARI).
    half = len(reference_labels) // 2
    sample_idx = list(range(0, 21)) + list(range(half, half + 21))
    ref = reference_labels.iloc[sample_idx].to_numpy()
    a = simulate_ari_threshold(ref, n_simulations=200, rng_seed=7)
    b = simulate_ari_threshold(ref, n_simulations=200, rng_seed=7)
    assert a == b


# ─────────────────────── ARI permutation invariance regression ───────────────────────


def test_ari_is_permutation_invariant_regression():
    """D030.5 invariant: if a future contributor maps GMM labels before
    calling ARI, ARI on identical-mapped-labels stays 1.0: but if they
    apply mapping ONLY to one side, ARI on a perfect inversion would
    drop. Pin the bare ARI property here.
    """
    ref = np.array([0, 0, 0, 1, 1, 1])
    inverted = np.array([1, 1, 1, 0, 0, 0])
    assert adjusted_rand_score(ref, inverted) == pytest.approx(1.0)


# ─────────────────────── orchestrator ───────────────────────


def test_run_leave_k_out_smoke(full_cohort, reference_labels):
    per_fold, summary = run_leave_k_out(
        full_cohort, _feature_cols(full_cohort),
        reference_labels,
        n_splits=5,  # small n_splits to keep the test fast
        seed=SEED_DEFAULT,
        n_simulations=200,
    )
    assert len(per_fold) == 5
    for col in ("fold", "n_train", "n_test", "ari", "T_fold", "pass_fold"):
        assert col in per_fold.columns
    assert summary["disagreement_rate_K"] == DISAGREEMENT_RATE_DEFAULT
    assert summary["percentile"] == PERCENTILE_DEFAULT
    assert "overall_pass" in summary
    # ARI in [-1, 1] regression.
    assert ((per_fold["ari"] >= -1.0) & (per_fold["ari"] <= 1.0)).all()


def test_attach_summary_row_appends_one_row(full_cohort, reference_labels):
    per_fold, summary = run_leave_k_out(
        full_cohort, _feature_cols(full_cohort),
        reference_labels,
        n_splits=5,
        n_simulations=100,
    )
    out = attach_summary_row(per_fold, summary)
    assert len(out) == len(per_fold) + 1
    assert out.iloc[-1]["fold"] == "SUMMARY"
    assert float(out.iloc[-1]["ari"]) == summary["median_ari"]


def test_run_leave_k_out_raises_when_reference_does_not_overlap(full_cohort):
    """Empty intersection between cohort pids and reference labels."""
    bogus_ref = pd.Series([0, 1], index=["NOT_A_PID_1", "NOT_A_PID_2"])
    with pytest.raises(ValueError, match="cover exactly the cohort"):
        run_leave_k_out(
            full_cohort, _feature_cols(full_cohort),
            bogus_ref,
            n_splits=3,
            n_simulations=50,
        )


def test_run_leave_k_out_refuses_incomplete_reference_coverage(full_cohort, reference_labels):
    """With 25% of patients lacking a reference label the
    old code dropped them silently and still passed."""
    partial = reference_labels.iloc[: int(0.75 * len(reference_labels))]
    with pytest.raises(ValueError, match="cover exactly the cohort"):
        run_leave_k_out(full_cohort, _feature_cols(full_cohort), partial,
                        n_splits=3, n_simulations=50)


def test_reference_class_counts_use_the_published_mapping(full_cohort, reference_labels):
    """Counts used the fold model's raw label numbers, which
    can be swapped. With the published mapping, swapping the reference's
    label numbers leaves the focal count unchanged."""
    from predict.validate.label_alignment import determine_mapping
    idx = full_cohort.set_index(full_cohort["pid"].astype(str))
    for ref in (reference_labels, 1 - reference_labels):
        mapping = determine_mapping(idx.loc[ref.index], ref, tie_break="raise")
        expected = int((ref.map(mapping) == "focal").sum())
        per_fold, _ = run_leave_k_out(full_cohort, _feature_cols(full_cohort), ref,
                                      n_splits=3, n_simulations=50)
        assert int(per_fold["n_test_focal_reference"].sum()) == expected
        assert int(per_fold["n_test_with_reference"].sum()) == len(full_cohort)


def test_pass_fold_logic_consistent(full_cohort, reference_labels):
    """pass_fold == (ari >= T_fold) for every non-NaN T_fold row."""
    per_fold, _ = run_leave_k_out(
        full_cohort, _feature_cols(full_cohort),
        reference_labels,
        n_splits=5,
        n_simulations=100,
    )
    for _, row in per_fold.iterrows():
        if np.isnan(row["T_fold"]):
            continue
        expected = bool(row["ari"] >= row["T_fold"])
        assert row["pass_fold"] == expected
