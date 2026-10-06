#!/usr/bin/env python
"""Start and publish the stage 9 exploratory lesion chain (D043).

    python scripts/09_stage9.py begin    # verify inputs, empty exploratory.partial
    PREDICT_STAGE9_ROOT=exploratory.partial python experiments/...   # the chain
    python scripts/09_stage9.py finish   # verify, record, publish

``finish`` re-verifies the inputs recorded by ``begin``, requires every file
in ``predict.stage9.REQUIRED_OUTPUTS``, checks the LAD termination records
(a matched comparison exists only where matching succeeded), writes
``COMPLETE_stage9.json`` and swaps the folder in. A failed chain never
reaches ``finish``, so the previous complete result stays published.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from predict.completion import write_json_atomic
from predict.config import load_config
from predict.provenance import code_revision
from predict.publish import fresh_partial, publish
from predict.stage9 import (
    BEGIN_FILE, REQUIRED_OUTPUTS, STAGE9_DIR, STAGE9_MARKER, STAGE9_PARTIAL,
    stage9_fingerprint, stage9_inputs, tree_manifest,
)

LAD_COHORT_DIRS = ("", "stratified_Qr36d_2", "stratified_I30f_3")


def check_lad_terminations(root: Path) -> None:
    """A matched comparison may exist only where matching succeeded."""
    for sub in LAD_COHORT_DIRS:
        d = root / "lad_phenotype" / sub if sub else root / "lad_phenotype"
        verdict = json.loads((d / "matched_diagnostics.json").read_text(encoding="utf-8"))["verdict"]
        has_cmp = (d / "matched_comparison_summary.json").exists()
        if (verdict == "match_ok") != has_cmp:
            raise SystemExit(f"hard stop: {d}: matching verdict {verdict!r} but matched comparison "
                             f"{'present' if has_cmp else 'missing'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("begin", "finish"))
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    target = cfg.paths.outputs / STAGE9_DIR
    partial = cfg.paths.outputs / STAGE9_PARTIAL

    if args.action == "begin":
        inputs = stage9_inputs(cfg)
        fresh_partial(target)
        write_json_atomic(partial / BEGIN_FILE, {
            "inputs": inputs, "stage9_fingerprint": stage9_fingerprint(),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"stage 9 started in {partial}")
        return 0

    if not (partial / BEGIN_FILE).exists():
        raise SystemExit(f"hard stop: {partial} was not started by `begin`")
    begin = json.loads((partial / BEGIN_FILE).read_text(encoding="utf-8"))
    inputs = stage9_inputs(cfg)
    if inputs != begin["inputs"] or stage9_fingerprint() != begin["stage9_fingerprint"]:
        raise SystemExit("hard stop: stage 9 inputs or code changed during the chain; nothing published")
    missing = [f for f in REQUIRED_OUTPUTS if not (partial / f).exists()]
    if missing:
        raise SystemExit(f"hard stop: the stage 9 chain is incomplete; missing {missing}")
    check_lad_terminations(partial)
    write_json_atomic(partial / STAGE9_MARKER, {
        "stage": 9,
        "stage9_fingerprint": begin["stage9_fingerprint"],
        "inputs": inputs,
        "outputs": tree_manifest(partial),
        "revision": code_revision(cfg.paths.outputs.parent),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    publish(partial, target)
    print(f"stage 9 published to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
