#!/usr/bin/env python
"""How much reconstruction-kernel signal reaches clustering, per gate root?

Motivation: under the D033 and T5 gates, kmeans forced k=3 failed the D021
kernel chi-square (p .042 / .035) while production D014 passed. ComBat
(D019) harmonised only a hardcoded 6-name list, so most gated radiomics
features entered clustering unharmonised; D035 fixed that. Which features a
run actually harmonised is read from its own matrix_prep_log.json
(combat_audit), so this works before and after the fix.

For each root's full-cohort 06_reduce/ this reports:
  1. per feature in prepared_matrix.csv: kernel R^2 (one-hot OLS, the same
     statistic as the ComBat audit), whether ComBat touched it, whether it is
     a representative; flags R^2 > 0.02 (the D019 acceptance bound);
  2. per PC: kernel R^2;
  3. kernel predictability: 5-fold stratified logistic regression AUC,
     10 repeats, from (a) all PCs and (b) the representative features, with
     a 200-permutation null (label shuffle) for a p-value.
Only the two main kernels are used for the AUC (singletons dropped).

Usage
-----
    python scripts/diagnose_kernel_leak.py production d014 d033 T5_excellent

Output: outputs_gates/kernel_leak/{features,pcs,auc}.csv and summary.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
R2_BOUND = 0.02
N_REPEATS = 10
N_PERM = 200


def _load_script_module(rel_path: str, alias: str):
    spec = importlib.util.spec_from_file_location(alias, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _kernel_r2(y: np.ndarray, kernel: pd.Series) -> float:
    from predict.reduce.prepare_matrix import _explained_variance_by_kernel
    df = pd.DataFrame({"y": y, "kernel": kernel.to_numpy()})
    return _explained_variance_by_kernel(df, ["y"], kernel_col="kernel")["y"]


def _cv_auc(X: np.ndarray, y: np.ndarray, seed: int, repeats: int = N_REPEATS) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + rep)
        p = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs))


def _auc_with_null(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    obs = _cv_auc(X, y, seed)
    rng = np.random.default_rng(seed)
    # One CV repeat per permutation keeps the null cheap.
    null = np.asarray([_cv_auc(X, rng.permutation(y), seed, repeats=1) for _ in range(N_PERM)])
    p = (1 + int((null >= obs).sum())) / (1 + len(null))
    return obs, float(np.percentile(null, 95)), p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs_gates" / "kernel_leak")
    args = ap.parse_args()

    cmp = _load_script_module("scripts/compare_gate_roots.py", "_cmp_roots")

    feat_rows, pc_rows, auc_rows = [], [], []
    for name in args.roots:
        red = cmp._root(name) / "06_reduce"
        prep = pd.read_csv(red / "prepared_matrix.csv", dtype={"pid": str})
        meta = pd.read_csv(red / "cohort_metadata.csv", dtype={"pid": str})
        kern = meta.set_index("pid").loc[prep["pid"], "kernel"].astype(str).reset_index(drop=True)
        reps = set(pd.read_csv(red / "representative_features.csv").iloc[:, 0])
        import json
        prep_log = json.loads((red / "matrix_prep_log.json").read_text(encoding="utf-8"))
        numeric = [c for c in prep_log["zscore_columns"] if c in prep.columns]
        harmonised = {r["feature"] for r in prep_log.get("combat_audit", [])
                      if not r.get("skipped")}
        for c in numeric:
            r2 = _kernel_r2(prep[c].to_numpy(float), kern)
            feat_rows.append({"root": name, "feature": c, "kernel_r2": r2,
                              "over_bound": r2 > R2_BOUND,
                              "radiomics": c.startswith("original_"),
                              "shape": c.startswith("original_shape_"),
                              "combat_harmonised": c in harmonised,
                              "representative": c in reps})

        pcs = pd.read_csv(red / "pca_scores.csv", dtype={"pid": str}).set_index("pid")
        k_pc = meta.set_index("pid").loc[pcs.index, "kernel"].astype(str).reset_index(drop=True)
        for c in pcs.columns:
            pc_rows.append({"root": name, "pc": c, "kernel_r2": _kernel_r2(pcs[c].to_numpy(float), k_pc)})

        main2 = k_pc.value_counts().index[:2]
        keep = k_pc.isin(main2).to_numpy()
        y = (k_pc[keep] == main2[0]).astype(int).to_numpy()
        auc_pc, null95_pc, p_pc = _auc_with_null(pcs.to_numpy(float)[keep], y)
        rep_cols = [c for c in numeric if c in reps]
        Xr = prep.set_index("pid").loc[pcs.index, rep_cols].to_numpy(float)[keep]
        auc_rep, null95_rep, p_rep = _auc_with_null(Xr, y)
        auc_rows.append({"root": name, "n": int(keep.sum()),
                         "auc_pcs": auc_pc, "null95_pcs": null95_pc, "p_pcs": p_pc,
                         "auc_reps": auc_rep, "null95_reps": null95_rep, "p_reps": p_rep})
        print(f"{name}: done", flush=True)

    feats, pcs_df, aucs = pd.DataFrame(feat_rows), pd.DataFrame(pc_rows), pd.DataFrame(auc_rows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out / "features.csv", index=False)
    pcs_df.to_csv(out / "pcs.csv", index=False)
    aucs.to_csv(out / "auc.csv", index=False)

    L = ["=" * 88, "KERNEL LEAK DIAGNOSTIC (prepared matrix after D019, full cohort)", "=" * 88]
    for name in args.roots:
        f = feats[feats.root == name]
        rad = f[f.radiomics & ~f["shape"]]
        a = aucs[aucs.root == name].iloc[0]
        L.append(f"\n== {name}")
        L.append(f"  non-shape radiomics in matrix {len(rad)}; ComBat-harmonised "
                 f"{int(rad.combat_harmonised.sum())}; unharmonised {int((~rad.combat_harmonised).sum())}")
        L.append(f"  features with kernel R2 > {R2_BOUND}: {int(f.over_bound.sum())} "
                 f"(representatives among them: {int((f.over_bound & f.representative).sum())})")
        top = f.sort_values("kernel_r2", ascending=False).head(8)
        for r in top.itertuples():
            L.append(f"    {r.kernel_r2:6.3f}  {r.feature.replace('original_', ''):<45}"
                     f"{'rep ' if r.representative else '    '}{'combat' if r.combat_harmonised else ''}")
        pcs_n = pcs_df[pcs_df.root == name].sort_values("kernel_r2", ascending=False).head(3)
        L.append("  top PCs by kernel R2: " + ", ".join(f"{r.pc} {r.kernel_r2:.3f}" for r in pcs_n.itertuples()))
        L.append(f"  kernel AUC from PCs  {a.auc_pcs:.3f} (null 95th pct {a.null95_pcs:.3f}, p {a.p_pcs:.3f})")
        L.append(f"  kernel AUC from reps {a.auc_reps:.3f} (null 95th pct {a.null95_reps:.3f}, p {a.p_reps:.3f})")
    L.append("=" * 88)
    text = "\n".join(L)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
