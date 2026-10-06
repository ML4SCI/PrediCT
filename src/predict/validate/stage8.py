"""Completion record for stage 8 (``outputs/08_validate/``).

Stage 8 refits the production spatial pipeline on the published cohort
(D029 holdout, D030 folds) and re-exports stage 7's cross-cohort ARI
(D031). Safeguards (D041):

* ``stage8_inputs`` requires a complete, current stage 7 record (which
  covers the stage 6 records, stage 5 files and features.csv) and that the
  gate files stage 8 refits with are the ones stage 5 recorded.
* 09_validate writes into ``08_validate.partial/`` and publishes the folder
  with ``COMPLETE_stage8.json`` only after every deliverable completed;
  reduced runs (skipped deliverables or non-production CV settings) go to
  ``outputs/08_validate_smoke/`` and are never published.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predict.analyse.stage7 import STAGE7_MARKER, require_stage7_complete
from predict.completion import code_sha, digest, file_sha, require_marker
from predict.config import REPO_ROOT, gate_dir
from predict.discover.stage6 import STAGE5_MARKER

STAGE8_MARKER = "COMPLETE_stage8.json"
_SCRIPTS = REPO_ROOT / "scripts"
# 09_validate reruns stage 2 and 3 processing on the holdout patients.
STAGE8_EXTRA = (_SCRIPTS / "09_validate.py", _SCRIPTS / "02_preprocess.py",
                _SCRIPTS / "03_features.py")
STAGE8_PACKAGES = ("validate", "analyse", "reduce", "discover", "io", "preprocess", "features")


def stage8_fingerprint() -> str:
    return digest({"code": code_sha(extra=STAGE8_EXTRA, packages=STAGE8_PACKAGES)})


def stage8_inputs(cfg: Any) -> dict:
    """Verify what stage 8 reads and return the input hashes it records."""
    outputs = cfg.paths.outputs
    require_stage7_complete(outputs / "07_analyse", cfg)
    rec5 = json.loads((outputs / "06_reduce" / STAGE5_MARKER).read_text(encoding="utf-8"))
    gates = {
        "gated_features_sha": file_sha(gate_dir(cfg) / "gated_features.csv"),
        "derived_icc_sha": file_sha(gate_dir(cfg) / "derived_icc.csv"),
    }
    stale = [k for k, v in gates.items() if rec5.get(k) != v]
    if stale:
        raise SystemExit(f"hard stop: gate files changed since stage 5 ran ({', '.join(stale)}); "
                         "rerun stages 5-7")
    return {"stage7_record_sha": file_sha(outputs / "07_analyse" / STAGE7_MARKER), **gates}


def require_stage8_complete(validate_dir: Path, cfg: Any) -> dict:
    """The stage 8 record if it matches the current inputs and code and
    every listed output is unchanged; otherwise stop."""
    if not (Path(validate_dir) / STAGE8_MARKER).exists():
        raise SystemExit(f"hard stop: {Path(validate_dir) / STAGE8_MARKER} missing; stage 8 "
                         "did not finish cleanly or never ran")
    marker = require_marker(validate_dir, {
        "stage8_fingerprint": stage8_fingerprint(),
        "inputs": stage8_inputs(cfg),
    }, name=STAGE8_MARKER)
    bad = [n for n, sha in marker["outputs"].items() if file_sha(Path(validate_dir) / n) != sha]
    if bad:
        raise SystemExit(f"hard stop: stage 8 files in {validate_dir} changed or missing "
                         f"since their completion record: {bad[:5]}")
    return marker


__all__ = ["STAGE8_MARKER", "stage8_fingerprint", "stage8_inputs", "require_stage8_complete"]
