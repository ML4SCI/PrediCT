"""Completion record for stage 7 (``outputs/07_analyse/``).

Stage 7 combines each cohort's stage 5-6 outputs with RAW measurements from
stage 3's ``features.csv``. Two safeguards:

* ``stage7_inputs`` verifies every cohort's stage 6 record and that
  ``features.csv`` is byte-identical to the file stage 5 read (its hash is
  in ``COMPLETE_stage5.json``), so a changed features.csv cannot silently
  change the reported medians.
* 08_analyse writes into ``07_analyse.partial/`` and publishes the folder
  with ``COMPLETE_stage7.json`` (inputs, settings, code fingerprint, output
  hashes) only after a complete run. Every consumer (the stage 7 report,
  the focal/diffuse verification, stage 8, the figures) calls
  :func:`require_stage7_complete`, the one implementation of the check.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predict.completion import code_sha, digest, file_sha, require_marker
from predict.config import REPO_ROOT
from predict.discover.stage6 import STAGE5_MARKER, STAGE6_MARKER, require_stage6_complete

STAGE7_MARKER = "COMPLETE_stage7.json"
STAGE7_SCRIPT = REPO_ROOT / "scripts" / "08_analyse.py"
# analyse, plus reduce.prepare_matrix (derived features) and
# discover.validity (partition ARI), which stage 7 imports.
STAGE7_PACKAGES = ("analyse", "reduce", "discover")

# Cohort label -> folder under outputs/06_reduce/ ("/" is not a valid name).
COHORT_DIRS: dict[str, str] = {
    "full": "",
    "Qr36d/2": "stratified_Qr36d_2",
    "I30f/3": "stratified_I30f_3",
}


def cohort_dir(reduce_root: Path, cohort: str) -> Path:
    sub = COHORT_DIRS[cohort]
    return Path(reduce_root) if sub == "" else Path(reduce_root) / sub


def features_csv_path(cfg: Any) -> Path:
    return cfg.paths.outputs / "03_features" / "features.csv"


def stage7_fingerprint() -> str:
    return digest({"code": code_sha(extra=[STAGE7_SCRIPT], packages=STAGE7_PACKAGES)})


def stage7_inputs(cfg: Any) -> dict:
    """Verify the inputs stage 7 reads and return their hashes.

    Stops unless every cohort's stage 6 record is complete and current and
    ``features.csv`` is the file stage 5 recorded for that cohort.
    """
    reduce_root = cfg.paths.outputs / "06_reduce"
    feats = features_csv_path(cfg)
    feats_sha = file_sha(feats)
    records: dict[str, str] = {}
    for cohort in COHORT_DIRS:
        cdir = cohort_dir(reduce_root, cohort)
        require_stage6_complete(cdir, cfg)
        recorded = json.loads((cdir / STAGE5_MARKER).read_text(encoding="utf-8")).get("features_csv_sha")
        if recorded != feats_sha:
            raise SystemExit(
                f"hard stop: {feats} is not the file stage 5 read for cohort {cohort} "
                f"(recorded {recorded or 'nothing'}, current {feats_sha[:16]}); "
                "rerun stages 5-6, or restore the original features.csv")
        records[cohort] = file_sha(cdir / STAGE6_MARKER)
    return {"features_csv_sha": feats_sha, "stage6_records": records}


def require_stage7_complete(analyse_dir: Path, cfg: Any) -> dict:
    """The stage 7 record if it matches the current inputs and stage 7 code
    and every listed output is unchanged; otherwise stop."""
    if not (Path(analyse_dir) / STAGE7_MARKER).exists():
        raise SystemExit(f"hard stop: {Path(analyse_dir) / STAGE7_MARKER} missing; stage 7 "
                         "did not finish cleanly or never ran")
    marker = require_marker(analyse_dir, {
        "stage7_fingerprint": stage7_fingerprint(),
        "inputs": stage7_inputs(cfg),
    }, name=STAGE7_MARKER)
    bad = [n for n, sha in marker["outputs"].items() if file_sha(Path(analyse_dir) / n) != sha]
    if bad:
        raise SystemExit(f"hard stop: stage 7 files in {analyse_dir} changed or missing "
                         f"since their completion record: {bad[:5]}")
    return marker


__all__ = [
    "COHORT_DIRS", "STAGE7_MARKER", "cohort_dir", "features_csv_path",
    "stage7_fingerprint", "stage7_inputs", "require_stage7_complete",
]
