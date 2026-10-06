"""Completion records that bind a cohort folder's stage 5 and stage 6 outputs.

A cohort folder (``outputs/06_reduce/`` or ``outputs/06_reduce/stratified_*``)
holds stage 5 outputs and, after stage 6, the discovery outputs read by
stages 7, 8 and the figures. Two records live in it:

* ``COMPLETE_stage5.json`` (written by 06_reduce.py): every stage 5 file and
  its hash.
* ``COMPLETE_stage6.json`` (written by 07_discover.py): the hash of the stage
  5 record it ran on, the stage 6 code/config fingerprint, the production
  run settings, and every stage 6 output file with its hash.

Downstream code calls :func:`require_stage6_complete` (one implementation,
so stages 7 and 8 cannot disagree): it refuses a missing record, a record
from other stage 5 inputs or other stage 6 code/config, and any listed file
whose hash changed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predict.completion import code_sha, digest, file_sha, require_marker
from predict.config import REPO_ROOT

STAGE5_MARKER = "COMPLETE_stage5.json"
STAGE6_MARKER = "COMPLETE_stage6.json"

# Production settings of 07_discover.py (D021). Any other value makes a smoke
# run. Stage 8 reads random_state from here to refit the production
# spatial PCA and GMM with the same settings.
PRODUCTION: dict = {
    "k_range": list(range(1, 13)),
    "gap_bootstraps": 500,
    "consensus_subsamples": 100,
    "hennig_bootstraps": 100,
    "random_state": 42,
    "forced_k": 3,
}
STAGE6_SCRIPT = REPO_ROOT / "scripts" / "07_discover.py"
# Code stage 6 runs (discovery, plus reduce.fit_pca for the spatial space)
# and the config sections it reads.
STAGE6_PACKAGES = ("discover", "reduce")
STAGE6_CONFIG_SECTIONS = ("seed", "reduce", "consensus", "gap_statistic",
                          "stability_cohort", "hopkins")


def stage6_fingerprint(cfg: Any) -> str:
    return digest({
        "code": code_sha(extra=[STAGE6_SCRIPT], packages=STAGE6_PACKAGES),
        "config": {k: cfg.raw.get(k) for k in STAGE6_CONFIG_SECTIONS},
    })


def _verify_files(folder: Path, files: dict[str, str], what: str) -> None:
    bad = [n for n, sha in files.items() if file_sha(Path(folder) / n) != sha]
    if bad:
        raise SystemExit(f"hard stop: {what} files in {folder} changed or missing "
                         f"since their completion record: {bad[:5]}")


def require_stage5_complete(cohort_dir: Path) -> dict:
    """The stage 5 record of ``cohort_dir``, with every listed file verified."""
    path = Path(cohort_dir) / STAGE5_MARKER
    if not path.exists():
        raise SystemExit(f"hard stop: {path} missing; rerun stage 5 for this cohort")
    marker = json.loads(path.read_text(encoding="utf-8"))
    _verify_files(cohort_dir, marker["files"], "stage 5")
    return marker


def require_stage6_complete(cohort_dir: Path, cfg: Any) -> dict:
    """The stage 6 record of ``cohort_dir`` if it matches the current stage 5
    record and stage 6 code/config and every listed output is unchanged."""
    marker = require_marker(cohort_dir, {
        "stage5_marker_sha": file_sha(Path(cohort_dir) / STAGE5_MARKER),
        "stage6_fingerprint": stage6_fingerprint(cfg),
    }, name=STAGE6_MARKER)
    require_stage5_complete(cohort_dir)
    _verify_files(cohort_dir, marker["outputs"], "stage 6")
    return marker


__all__ = [
    "PRODUCTION", "STAGE5_MARKER", "STAGE6_MARKER", "stage6_fingerprint",
    "require_stage5_complete", "require_stage6_complete",
]
