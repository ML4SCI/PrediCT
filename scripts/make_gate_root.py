#!/usr/bin/env python
"""Build an isolated outputs root so stages 6-9 run on a chosen feature gate.

Stages 6-9 read their inputs from fixed subdirectories of ``paths.outputs``
(03_features/, 05_icc/, 06_reduce/, 07_analyse/, ...). Rather than editing
each stage, this script builds ``outputs_gates/<name>/`` containing:

  01_manifest, 02_preprocessed, 03_features   symlinks to the shared outputs/
  05_icc/icc_report.csv, gated_features.csv   copied from the chosen gate
  05_icc/SOURCE.txt                           provenance
  config.yaml                                 configs/default.yaml with
                                              paths.outputs pointed here

Stages 6-9 then run unchanged with ``--config outputs_gates/<name>/config.yaml``
and write their own 06_reduce/, 07_analyse/, 08_validate/ inside the root.
The production outputs/ tree is never written.

Gates:
    d033               outputs/05b_icc_realistic/          (D033 primary, T1)
    T1_primary .. T5_excellent
                       outputs/05c_robustness_tiers/<tier>/ (D034 tiers)
    d014               outputs/05_icc/                      (supplementary gate, for comparison)

Usage
-----
    python scripts/make_gate_root.py d033
    python scripts/make_gate_root.py T5_excellent
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED = ("01_manifest", "02_preprocessed", "03_features")
TIERS = ("T1_primary", "T2_ci", "T3_zshift", "T4_all_arms", "T5_excellent")


def gate_source(outputs: Path, gate: str) -> Path:
    if gate == "d033":
        return outputs / "05b_icc_realistic"
    if gate == "d014":
        return outputs / "05_icc"
    if gate in TIERS:
        return outputs / "05c_robustness_tiers" / gate
    raise SystemExit(f"unknown gate {gate!r}; use d033, d014 or one of {TIERS}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate")
    ap.add_argument("--name", default=None, help="root name (default: the gate name)")
    ap.add_argument("--root", default=None,
                    help="root directory relative to the repo (default: outputs_gates/<name>)")
    args = ap.parse_args()

    from predict.config import DEFAULT_CONFIG_PATH, load_config
    cfg = load_config()
    outputs = cfg.paths.outputs
    src = gate_source(outputs, args.gate)
    for f in ("icc_report.csv", "gated_features.csv"):
        if not (src / f).exists():
            raise SystemExit(f"missing {src / f}")
    if (src.parent / "PARTIAL_SMOKE_DO_NOT_USE").exists() or (src / "PARTIAL_SMOKE_DO_NOT_USE").exists():
        raise SystemExit(f"{src} is a partial smoke output")

    name = args.name or args.gate
    rel_root = Path(args.root) if args.root else Path("outputs_gates") / name
    root = REPO_ROOT / rel_root
    if rel_root.is_absolute() or root.resolve() == outputs.resolve():
        raise SystemExit(f"--root must be a new directory relative to the repo, not {rel_root}")
    root.mkdir(parents=True, exist_ok=True)

    for d in SHARED:
        link, target = root / d, outputs / d
        if not target.exists():
            raise SystemExit(f"missing shared input {target}")
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            raise SystemExit(f"{link} exists and is not the expected symlink; remove it first")
        link.symlink_to(target.resolve(), target_is_directory=True)

    icc_dir = root / "05_icc"
    icc_dir.mkdir(exist_ok=True)
    for f in ("icc_report.csv", "gated_features.csv"):
        shutil.copy2(src / f, icc_dir / f)
    # D040: a D033-based gate (5b or a 05c tier) carries the measured
    # reliability of the stage-5 transformed features; stage 6 needs it.
    if args.gate != "d014":
        dicc = outputs / "05b_icc_realistic" / "derived_icc.csv"
        if not dicc.exists():
            raise SystemExit(f"missing {dicc}; rerun stage 5b")
        shutil.copy2(dicc, icc_dir / "derived_icc.csv")
    n_gated = len(pd.read_csv(icc_dir / "gated_features.csv"))
    (icc_dir / "SOURCE.txt").write_text(
        f"gate {args.gate}\nsource {src}\ngated features {n_gated}\n", encoding="utf-8")

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"]["outputs"] = str(rel_root).replace("\\", "/")
    # The chosen gate is copied into 05_icc/ above, the "classic" layout.
    raw["stability"]["primary_gate"] = "classic"
    (root / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    print(f"gate root {rel_root}: {n_gated} gated features from {src}")
    print(f"run stages 6-9 with --config {rel_root}/config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
