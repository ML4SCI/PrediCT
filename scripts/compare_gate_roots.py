#!/usr/bin/env python
"""Compare stages 6-9 across gate roots built by make_gate_root.py.

For each root (``outputs_gates/<name>``, ``production`` for outputs/, or a
top-level ``outputs_*`` directory)
prints the headline numbers of every downstream stage, then agreement with
a reference root: identical representative sets, and ARI of the forced
k=3 labels (per algorithm) and of the spatial-only k=2 labels.

Usage
-----
    python scripts/compare_gate_roots.py production d014 d033 T5_excellent

The first root is the reference. Running the d014 root against production
is the harness control: it must agree exactly, otherwise the gate-root
mechanism itself changes results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _root(name: str) -> Path:
    """production -> outputs/; a top-level outputs_* dir as is;
    anything else -> outputs_gates/<name>."""
    if name == "production":
        return REPO_ROOT / "outputs"
    if name.startswith("outputs_") and (REPO_ROOT / name).is_dir():
        return REPO_ROOT / name
    return REPO_ROOT / "outputs_gates" / name


def _read(path: Path, kind: str = "csv"):
    if not path.exists():
        return None
    if kind == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def headline(name: str) -> dict:
    r = _root(name)
    red = r / "06_reduce"
    out: dict = {"root": name}
    from predict.config import gate_dir, load_config
    cfg_path = r / "config.yaml"
    g = _read(gate_dir(load_config(cfg_path if cfg_path.exists() else None)) / "gated_features.csv")
    out["gated"] = None if g is None else len(g)
    rep = _read(red / "representative_features.csv")
    out["representatives"] = None if rep is None else len(rep)
    pca = _read(red / "pca_scores.csv")
    out["pcs"] = None if pca is None else pca.shape[1] - 1
    hop = _read(red / "hopkins.json", "json")
    out["hopkins_H"] = None if hop is None else round(float(hop.get("H", float("nan"))), 3)
    gap = _read(red / "gap_statistic_summary.csv")
    if gap is not None:
        for row in gap.itertuples():
            out[f"k_{row.feature_space}_{row.algorithm}"] = int(row.selected_k)
    cons = _read(red / "consensus_summary.json", "json")
    if cons:
        for c in cons:
            out[f"pac_{c['algorithm']}"] = round(float(c["pac_score"]), 3)
    val = _read(red / "validity_checks.csv")
    if val is not None:
        h = val[(val["test"] == "hennig_clusterboot") & (val.get("feature_space", "full") == "full")]
        out["hennig_stable_full"] = f"{int(h['stable'].sum())}/{len(h)}"
        hs = val[(val["test"] == "hennig_clusterboot") & (val.get("feature_space") == "spatial_only")]
        out["hennig_stable_spatial_k2"] = f"{int(hs['stable'].sum())}/{len(hs)}"
        chi = val[val["test"] == "kernel_chi_square"]
        out["kernel_chi2_pass"] = f"{int(chi['passes'].astype(bool).sum())}/{len(chi)}"
    verdict = _read(r / "07_analyse" / "directional_verdict.json", "json")
    out["directional_verdict"] = None if verdict is None else json.dumps(verdict, sort_keys=True)[:200]
    out["validate_files"] = len(list((r / "08_validate").glob("*.csv"))) if (r / "08_validate").exists() else 0
    return out


def agreement(ref: str, other: str) -> dict:
    from sklearn.metrics import adjusted_rand_score as ari
    a, b = _root(ref) / "06_reduce", _root(other) / "06_reduce"
    out: dict = {"root": other, "vs": ref}
    ra, rb = _read(a / "representative_features.csv"), _read(b / "representative_features.csv")
    if ra is not None and rb is not None:
        col = "feature" if "feature" in ra.columns else ra.columns[0]
        sa, sb = set(ra[col]), set(rb[col])
        out["reps_identical"] = sa == sb
        out["reps_shared"] = len(sa & sb)
    for fname, prefix in (("cluster_labels_forced.csv", "ari_forced_"),
                          ("cluster_labels_spatial_k2.csv", "ari_")):
        la, lb = _read(a / fname), _read(b / fname)
        if la is None or lb is None:
            continue
        m = la.merge(lb, on="pid", suffixes=("_a", "_b"))
        for c in [c for c in la.columns if c != "pid"]:
            if f"{c}_a" in m and f"{c}_b" in m:
                out[prefix + c] = round(float(ari(m[f"{c}_a"], m[f"{c}_b"])), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs_gates" / "comparison.txt")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)

    heads = pd.DataFrame([headline(n) for n in args.roots]).set_index("root").T
    print("=" * 90)
    print("DOWNSTREAM HEADLINES PER GATE")
    print("=" * 90)
    print(heads.to_string())
    print()
    ref = args.roots[0]
    agree = pd.DataFrame([agreement(ref, n) for n in args.roots[1:]]).set_index("root").T
    print(f"AGREEMENT WITH {ref}")
    print(agree.to_string())
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(heads.to_string() + "\n\n" + agree.to_string() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
