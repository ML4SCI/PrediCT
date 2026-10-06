#!/usr/bin/env python
"""Where does the forced k=3 x kernel chi-square fail, per gate root?

After D035 every gated non-shape radiomics feature is ComBat-harmonised and
the PC space carries no detectable kernel signal (`diagnose_kernel_leak.py`),
yet one algorithm still fails the D021 kernel chi-square under
D033 and T5 while D014 passes. This script pinpoints the cause. Nothing here
changes a pipeline output; every refit is a diagnostic.

Per root (full cohort, 06_reduce/ seam) and algorithm (kmeans, ward, gmm):
  1. reproduce: refit forced k=3 at random_state 42 on pca_scores.npy, check
     ARI 1.0 against cluster_labels_forced.csv, report chi2 / p / Cramer's V
     and the contingency table with standardised residuals;
  2. seeds: refit at N seeds (kmeans, gmm; ward is deterministic) and report
     the p distribution and the fraction failing (is the fail seed luck?);
  3. burden: likelihood-ratio test of kernel ~ cluster + log1p(agatston)
     against kernel ~ log1p(agatston) (does the association survive burden?),
     plus kernel vs burden and category directly (population difference?);
  4. PCs: per-PC kernel R^2 and cluster eta^2; leave-one-PC-out refits; a
     refit on kernel-residualised PCs;
  5. representatives: re-PCA check against the seam, then leave-one-rep-out
     and group drops (canonical LCx, all canonical, all radiomics) with re-PCA
     at the same cumvar threshold, plus a refit with kernel regressed out of
     the canonical representatives only;
  6. stratified: ARI between the kernel-stratified rerun's forced k=3 labels
     and the full-cohort labels restricted to that kernel.

Roots run in parallel (one worker each, from default_n_jobs()).
    mkdir -p logs && qsub submit_kernel_chi2.sh
    OMP_NUM_THREADS=1 python scripts/diagnose_kernel_chi2.py d033 --n-seeds 20   # quick look

Output: outputs_gates/kernel_chi2/*.csv and summary.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from predict.config import default_n_jobs, load_config
from predict.discover.cluster_discovery import fit_cluster
from predict.discover.validity import kernel_association_given_burden, kernel_chi_square
from predict.reduce.pca import fit_pca

REPO_ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("kmeans", "ward", "gmm")
FORCED_K = 3
SEED = 42
ALPHA = 0.05


def _load_script_module(rel_path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def _chi(labels: np.ndarray, kern: np.ndarray):
    r = kernel_chi_square(labels, kern, alpha=ALPHA)
    n = r.contingency.sum()
    v = float(np.sqrt(r.chi2 / (n * (min(r.contingency.shape) - 1))))
    return r, v


def _onehot(kern: np.ndarray) -> np.ndarray:
    levels = sorted(set(kern))
    return np.column_stack([(kern == lv).astype(float) for lv in levels[1:]])


def _eta2(y: np.ndarray, groups: np.ndarray) -> float:
    """Share of variance in y explained by group means (= one-hot OLS R^2)."""
    y = np.asarray(y, float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_b = sum(((groups == g).sum() * (y[groups == g].mean() - y.mean()) ** 2)
               for g in np.unique(groups))
    return float(ss_b / ss_tot)


def _residualise(X: np.ndarray, kern: np.ndarray) -> np.ndarray:
    D = np.column_stack([np.ones(len(kern)), _onehot(kern)])
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ beta


def _refit_p(X: np.ndarray, algo: str, kern: np.ndarray, ref: np.ndarray) -> dict:
    lab = fit_cluster(X, k=FORCED_K, algorithm=algo, random_state=SEED)
    r, v = _chi(lab, kern)
    return {"chi2": r.chi2, "pval": r.pval, "cramers_v": v, "passes": r.passes,
            "ari_to_original": _ari(lab, ref)}


def analyse_root(name: str, red: Path, cumvar: float, n_seeds: int) -> dict[str, list[dict]]:
    X = np.load(red / "pca_scores.npy")
    pid_order = pd.read_csv(red / "pca_scores_pid_order.csv", dtype={"pid": str})["pid"].tolist()
    meta = pd.read_csv(red / "cohort_metadata.csv", dtype={"pid": str}).set_index("pid").loc[pid_order]
    kern = meta["kernel"].astype(str).to_numpy()
    prep = pd.read_csv(red / "prepared_matrix.csv", dtype={"pid": str})
    reps = pd.read_csv(red / "representative_features.csv").iloc[:, 0].tolist()
    stored = pd.read_csv(red / "cluster_labels_forced.csv", dtype={"pid": str}).set_index("pid").loc[pid_order]
    out: dict[str, list[dict]] = {k: [] for k in
                                  ("reproduce", "contingency", "seeds", "burden", "population",
                                   "pcs", "loo_pc", "loo_rep", "groups", "reps", "stratified")}

    # 3a. population: does kernel differ in burden / category regardless of clusters?
    from scipy.stats import chi2 as chi2_dist, chi2_contingency, mannwhitneyu
    main2 = pd.Series(kern).value_counts().index[:2].tolist()
    keep = np.isin(kern, main2)
    burden = meta["agatston_total"].to_numpy(float)
    a, b = burden[kern == main2[0]], burden[kern == main2[1]]
    out["population"].append({"root": name, "test": "agatston_by_kernel",
                              "kernel_a": main2[0], "n_a": len(a), "median_a": float(np.median(a)),
                              "kernel_b": main2[1], "n_b": len(b), "median_b": float(np.median(b)),
                              "pval": float(mannwhitneyu(a, b).pvalue)})
    if "category" in meta.columns:
        ct = pd.crosstab(meta["category"].astype(str).to_numpy()[keep], kern[keep])
        out["population"].append({"root": name, "test": "category_by_kernel",
                                  "pval": float(chi2_contingency(ct.to_numpy())[1]),
                                  "table": ct.to_dict()})

    labels: dict[str, np.ndarray] = {}
    for algo in ALGORITHMS:
        # 1. reproduce
        lab = fit_cluster(X, k=FORCED_K, algorithm=algo, random_state=SEED)
        labels[algo] = lab
        r, v = _chi(lab, kern)
        out["reproduce"].append({"root": name, "algorithm": algo, "chi2": r.chi2, "pval": r.pval,
                                 "dof": r.dof, "cramers_v": v, "passes": r.passes,
                                 "ari_vs_stored": _ari(lab, stored[f"forced_k{FORCED_K}_{algo}"].to_numpy()),
                                 "sizes": np.bincount(lab).tolist()})
        resid = (r.contingency - r.expected) / np.sqrt(r.expected)
        for i, c in enumerate(r.cluster_names):
            for j, k in enumerate(r.kernel_names):
                sub = burden[lab == int(c)]
                out["contingency"].append({"root": name, "algorithm": algo, "cluster": c, "kernel": k,
                                           "observed": int(r.contingency[i, j]),
                                           "expected": float(r.expected[i, j]),
                                           "std_residual": float(resid[i, j]),
                                           "cluster_median_agatston": float(np.median(sub))})

        # 2. seeds
        if algo != "ward":
            for s in range(n_seeds):
                ls = fit_cluster(X, k=FORCED_K, algorithm=algo, random_state=s)
                rs, vs = _chi(ls, kern)
                out["seeds"].append({"root": name, "algorithm": algo, "seed": s, "pval": rs.pval,
                                     "cramers_v": vs, "ari_to_seed42": _ari(ls, lab)})

        # 3b. burden-adjusted LR test (two main kernels); the library
        # implementation D021.1 uses in stage 6.
        adj = kernel_association_given_burden(lab, kern, burden)
        out["burden"].append({"root": name, "algorithm": algo,
                              **{k: v for k, v in adj.items() if k != "kernels"}})

        # 4. PCs (leave-one-out refits; per-PC table after the loop)
        for j in range(X.shape[1]):
            out["loo_pc"].append({"root": name, "algorithm": algo, "dropped": f"PC{j + 1}",
                                  **_refit_p(np.delete(X, j, axis=1), algo, kern, lab)})
        out["loo_pc"].append({"root": name, "algorithm": algo, "dropped": "KERNEL_RESIDUALISED_PCS",
                              **_refit_p(_residualise(X, kern), algo, kern, lab)})

    for j in range(X.shape[1]):
        out["pcs"].append({"root": name, "pc": f"PC{j + 1}", "kernel_r2": _eta2(X[:, j], kern),
                           **{f"cluster_eta2_{a_}": _eta2(X[:, j], labels[a_]) for a_ in ALGORITHMS}})

    # 5. representatives: re-PCA must reproduce the seam before drops mean anything
    prep_o = prep.set_index("pid").loc[pid_order].reset_index()
    base = fit_pca(prep_o, reps, cumvar_threshold=cumvar, random_state=SEED)
    repro = float(np.max(np.abs(base.scores - X))) if base.scores.shape == X.shape else float("inf")
    canonical = [c for c in reps if not c.startswith("original_")]
    groups = {"canonical_lcx": [c for c in canonical if "_lcx" in c],
              "all_canonical": canonical,
              "all_radiomics": [c for c in reps if c.startswith("original_")]}
    for c in reps:
        xc = prep_o[c].to_numpy(float)
        sd = xc.std(ddof=1)
        out["reps"].append({"root": name, "feature": c, "canonical": c in canonical,
                            "kernel_r2": _eta2(xc, kern),
                            "kernel_smd": float((xc[kern == main2[0]].mean() - xc[kern == main2[1]].mean()) / sd)
                            if sd > 0 else 0.0,
                            **{f"cluster_eta2_{a_}": _eta2(xc, labels[a_]) for a_ in ALGORITHMS}})

    def _repca_p(cols: list[str], frame: pd.DataFrame, algo: str) -> dict:
        p = fit_pca(frame, cols, cumvar_threshold=cumvar, random_state=SEED)
        return {"n_pcs": p.n_retain, **_refit_p(p.scores, algo, kern, labels[algo])}

    resid_frame = prep_o.copy()
    if canonical:
        resid_frame[canonical] = _residualise(prep_o[canonical].to_numpy(float), kern)
    for algo in ALGORITHMS:
        out["groups"].append({"root": name, "algorithm": algo, "dropped": "NONE (re-PCA)",
                              "seam_max_abs_diff": repro, **_repca_p(reps, prep_o, algo)})
        for g, cols in groups.items():
            rest = [c for c in reps if c not in cols]
            if cols and len(rest) >= 2:
                out["groups"].append({"root": name, "algorithm": algo, "dropped": g,
                                      "n_dropped": len(cols), **_repca_p(rest, prep_o, algo)})
        if canonical:
            out["groups"].append({"root": name, "algorithm": algo,
                                  "dropped": "KERNEL_RESIDUALISED_CANONICAL",
                                  "n_dropped": 0, **_repca_p(reps, resid_frame, algo)})
        for c in reps:
            out["loo_rep"].append({"root": name, "algorithm": algo, "dropped": c,
                                   **_repca_p([x for x in reps if x != c], prep_o, algo)})

    # 6. stratified reruns
    for sub in sorted(red.glob("stratified_*")):
        f = sub / "cluster_labels_forced.csv"
        if not f.exists():
            continue
        st = pd.read_csv(f, dtype={"pid": str}).set_index("pid")
        shared = [p for p in pid_order if p in st.index]
        idx = [pid_order.index(p) for p in shared]
        for algo in ALGORITHMS:
            col = f"forced_k{FORCED_K}_{algo}"
            if col in st.columns:
                out["stratified"].append({"root": name, "stratum": sub.name, "algorithm": algo,
                                          "n_shared": len(shared),
                                          "ari_full_vs_stratified": _ari(labels[algo][idx],
                                                                         st.loc[shared, col].to_numpy())})
    return out


def summarise(tabs: dict[str, pd.DataFrame], roots: list[str]) -> str:
    L = ["=" * 92, f"KERNEL CHI-SQUARE DIAGNOSTIC (forced k={FORCED_K}, full cohort)", "=" * 92]
    for name in roots:
        L.append(f"\n== {name}")
        pop = tabs["population"][tabs["population"].root == name]
        for r in pop.itertuples():
            if r.test == "agatston_by_kernel":
                L.append(f"  agatston by kernel: {r.kernel_a} n={r.n_a} median {r.median_a:.0f} vs "
                         f"{r.kernel_b} n={r.n_b} median {r.median_b:.0f}  (MWU p {r.pval:.3g})")
            else:
                L.append(f"  category by kernel: p {r.pval:.3g}")
        g = tabs["groups"]
        seam = g[(g.root == name) & (g.dropped == "NONE (re-PCA)")]["seam_max_abs_diff"]
        L.append(f"  re-PCA vs seam max |diff|: {seam.iloc[0]:.2e}" if len(seam) else "")
        for algo in ALGORITHMS:
            rp = tabs["reproduce"].query("root == @name and algorithm == @algo").iloc[0]
            L.append(f"  [{algo}] p {rp.pval:.3f} chi2 {rp.chi2:.2f} V {rp.cramers_v:.3f} "
                     f"{'PASS' if rp.passes else 'FAIL'}  sizes {rp.sizes}  ARI vs stored {rp.ari_vs_stored:.3f}")
            ct = tabs["contingency"].query("root == @name and algorithm == @algo")
            worst = ct.loc[ct.std_residual.abs().idxmax()]
            L.append(f"      largest residual: cluster {worst.cluster} x {worst.kernel} "
                     f"obs {worst.observed} exp {worst.expected:.1f} (z {worst.std_residual:+.2f}, "
                     f"cluster median agatston {worst.cluster_median_agatston:.0f})")
            sd = tabs["seeds"].query("root == @name and algorithm == @algo") if len(tabs["seeds"]) else None
            if sd is not None and len(sd):
                L.append(f"      seeds n={len(sd)}: fail {100 * (sd.pval < ALPHA).mean():.0f}%, p median "
                         f"{sd.pval.median():.3f} [5-95% {sd.pval.quantile(.05):.3f}-{sd.pval.quantile(.95):.3f}], "
                         f"ARI to seed 42 median {sd.ari_to_seed42.median():.2f}")
            bu = tabs["burden"].query("root == @name and algorithm == @algo").iloc[0]
            L.append(f"      LR kernel~cluster p {bu.p_cluster_unadjusted:.3f}; given burden "
                     f"p {bu.p_cluster_given_burden:.3f}; burden alone p {bu.p_burden_alone:.3g}")
            lp = tabs["loo_pc"].query("root == @name and algorithm == @algo")
            kr = lp[lp.dropped == "KERNEL_RESIDUALISED_PCS"].iloc[0]
            L.append(f"      kernel-residualised PCs: p {kr.pval:.3f} (ARI to original {kr.ari_to_original:.2f})")
            top = lp[lp.dropped != "KERNEL_RESIDUALISED_PCS"].sort_values("pval", ascending=False).head(3)
            L.append("      drop-one PC, highest p: " + ", ".join(
                f"{r.dropped} {r.pval:.3f} (ARI {r.ari_to_original:.2f})" for r in top.itertuples()))
            gg = g.query("root == @name and algorithm == @algo")
            L.append("      re-PCA drops: " + ", ".join(
                f"{r.dropped} p {r.pval:.3f}" for r in gg.itertuples()))
            lr = tabs["loo_rep"].query("root == @name and algorithm == @algo")
            top = lr.sort_values("pval", ascending=False).head(3)
            L.append("      drop-one rep, highest p: " + ", ".join(
                f"{r.dropped.replace('original_', '')} {r.pval:.3f} (ARI {r.ari_to_original:.2f})"
                for r in top.itertuples()))
            flips = lr[lr.passes != rp.passes]
            L.append(f"      drop-one rep flips the verdict: {len(flips)}/{len(lr)}")
        st = tabs["stratified"]
        if len(st) and (st.root == name).any():
            L.append("  stratified ARI (full labels within kernel vs stratified rerun): " + ", ".join(
                f"{r.stratum.replace('stratified_', '')}/{r.algorithm} {r.ari_full_vs_stratified:.2f}"
                for r in st[st.root == name].itertuples()))
        rp_ = tabs["reps"].query("root == @name").sort_values("kernel_r2", ascending=False).head(5)
        L.append("  reps by kernel R2: " + ", ".join(
            f"{r.feature.replace('original_', '')} {r.kernel_r2:.3f}" for r in rp_.itertuples()))
    L.append("=" * 92)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["production", "d014", "d033", "T5_excellent"])
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs_gates" / "kernel_chi2")
    args = ap.parse_args()

    from joblib import Parallel, delayed
    cmp = _load_script_module("scripts/compare_gate_roots.py", "_cmp_roots")
    cumvar = load_config(args.config).reduce.pca_cumvar
    n_jobs = min(len(args.roots), default_n_jobs())
    results = Parallel(n_jobs=n_jobs)(
        delayed(analyse_root)(name, cmp._root(name) / "06_reduce", cumvar, args.n_seeds)
        for name in args.roots)
    acc: dict[str, list[dict]] = {}
    for name, res in zip(args.roots, results):
        for k, rows in res.items():
            acc.setdefault(k, []).extend(rows)
        print(f"{name}: done", flush=True)

    tabs = {k: pd.DataFrame(v) for k, v in acc.items()}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for k, df in tabs.items():
        df.to_csv(out / f"{k}.csv", index=False)
    text = summarise(tabs, args.roots)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
