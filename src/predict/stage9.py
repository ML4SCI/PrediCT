"""Stage 9 (exploratory lesion chain): one root, one completion record.

The chain (lesion morphology, k sensitivity, C8 analyses, LAD phenotype, LM
isolated, lesion stability) writes into ``outputs/exploratory.partial/``
and is published to ``outputs/exploratory/`` as a whole, with
``COMPLETE_stage9.json``, only after every step completed (D043), so a
terminated step can never leave an earlier result in place for later steps
to read.

* ``scripts/09_stage9.py begin`` verifies the inputs and creates an empty
  ``exploratory.partial``; ``finish`` re-verifies them, checks the chain's
  required outputs and its termination records, writes the record and
  publishes.
* Every chain script resolves its root with :func:`stage9_root`, from
  ``PREDICT_STAGE9_ROOT`` (set by run_pipeline.sh to
  ``exploratory.partial``). The published folder is read-only: a script
  pointed at it refuses to run once it holds a completion record.
* Consumers (the figures) call :func:`require_stage9_complete`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from predict.completion import code_sha, digest, file_sha
from predict.config import REPO_ROOT

STAGE9_DIR = "exploratory"
STAGE9_PARTIAL = STAGE9_DIR + ".partial"
ROOT_ENV = "PREDICT_STAGE9_ROOT"
STAGE9_MARKER = "COMPLETE_stage9.json"
BEGIN_FILE = "STAGE9_BEGIN.json"

_EXPERIMENT_DIRS = ("lesion_morphology", "lad_phenotype", "lm_isolated_low_burden")
STAGE9_PACKAGES = ("analyse", "discover", "reduce", "features", "io", "stability")
# The chain imports one module of the figures package (the lesion class
# rules), so only that module is hashed and figure edits do not invalidate
# stage 9 (D044). figures/__init__.py is empty.
STAGE9_EXTRA_MODULES = ("figures/__init__.py", "figures/_lesion_classes.py")

# Files a complete chain must have produced (relative to the root).
REQUIRED_OUTPUTS = (
    "lesion_morphology/lesion_features.csv",
    "lesion_morphology/lesion_cluster_labels.csv",
    "lesion_morphology/cluster_profiles.csv",
    "lesion_morphology/cluster_vessel_chi_square.csv",
    "lesion_morphology/lesion_stability_d033/summary.txt",
    "lad_phenotype/lad_cluster_signature.json",
    "lad_phenotype/matched_diagnostics.json",
    "lad_phenotype/cross_stratum_replication.json",
    "lad_phenotype/stratified_Qr36d_2/lad_cluster_signature.json",
    "lad_phenotype/stratified_I30f_3/lad_cluster_signature.json",
    "lad_phenotype/stratified_Qr36d_2/matched_diagnostics.json",
    "lad_phenotype/stratified_I30f_3/matched_diagnostics.json",
    "lm_isolated_low_burden/summary.json",
    "lm_isolated_low_burden/cluster_overlap.json",
)


def _chain_files() -> list[Path]:
    out: list[Path] = [REPO_ROOT / "scripts" / "09_stage9.py"]
    out += [REPO_ROOT / "src" / "predict" / m for m in STAGE9_EXTRA_MODULES]
    for d in _EXPERIMENT_DIRS:
        root = REPO_ROOT / "experiments" / d
        out += sorted(p for p in root.glob("*.py"))
        out += sorted(root.glob("plan.md"))
    return out


def stage9_fingerprint() -> str:
    return digest({"code": code_sha(extra=_chain_files(), packages=STAGE9_PACKAGES)})


def stage9_root(cfg: Any) -> Path:
    """The folder the chain reads and writes (see the module docstring)."""
    name = os.environ.get(ROOT_ENV, STAGE9_DIR)
    if name not in (STAGE9_DIR, STAGE9_PARTIAL):
        raise SystemExit(f"{ROOT_ENV}={name!r}: allowed values are {STAGE9_DIR!r} and {STAGE9_PARTIAL!r}")
    root = cfg.paths.outputs / name
    if name == STAGE9_DIR and (root / STAGE9_MARKER).exists():
        raise SystemExit(
            f"{root} holds a published stage 9 result and is read-only; run the chain "
            f"through run_pipeline.sh, or `python scripts/09_stage9.py begin` and "
            f"{ROOT_ENV}={STAGE9_PARTIAL}")
    if name == STAGE9_PARTIAL and not (root / BEGIN_FILE).exists():
        raise SystemExit(f"{root} was not started; run `python scripts/09_stage9.py begin` first")
    return root


def stage9_inputs(cfg: Any) -> dict:
    """Verify what the chain reads and return the input hashes it records."""
    from predict.analyse.stage7 import STAGE7_MARKER, require_stage7_complete

    out = cfg.paths.outputs
    require_stage7_complete(out / "07_analyse", cfg)   # features.csv, stages 5-6
    calib = out / "04b_calibration" / "calibration.json"
    rec4b = out / "04b_realistic" / "COMPLETE.json"
    if not rec4b.exists():
        raise SystemExit(f"hard stop: {rec4b} missing; lesion stability needs the stage 4b calibration")
    if json.loads(rec4b.read_text(encoding="utf-8")).get("calibration_sha") != file_sha(calib):
        raise SystemExit(f"hard stop: {calib} is not the calibration stage 4b recorded")
    lesions = out / "03_features" / "lesions.csv"
    trips = out / "02_preprocessed" / "xml_roundtrip.csv"
    for p in (lesions, trips):
        if not p.exists():
            raise SystemExit(f"hard stop: {p} missing")
    return {
        "stage7_record_sha": file_sha(out / "07_analyse" / STAGE7_MARKER),
        "lesions_csv_sha": file_sha(lesions),
        "xml_roundtrip_sha": file_sha(trips),
        "calibration_sha": file_sha(calib),
    }


def tree_manifest(root: Path) -> dict[str, str]:
    """{relative path: sha256} of every file under ``root`` except the record."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): file_sha(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != STAGE9_MARKER}


def require_stage9_complete(cfg: Any) -> dict:
    """The published stage 9 record if it matches the current inputs and
    chain code and the folder holds exactly the recorded files."""
    root = cfg.paths.outputs / STAGE9_DIR
    path = root / STAGE9_MARKER
    if not path.exists():
        raise SystemExit(f"hard stop: {path} missing; stage 9 did not finish cleanly or never ran")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker.get("stage9_fingerprint") != stage9_fingerprint():
        raise SystemExit(f"hard stop: {path} was written by other stage 9 code; rerun stage 9")
    if marker.get("inputs") != stage9_inputs(cfg):
        raise SystemExit(f"hard stop: {path} does not match the current inputs; rerun stage 9")
    if tree_manifest(root) != marker["outputs"]:
        raise SystemExit(f"hard stop: files in {root} changed, were added or are missing since "
                         "their completion record")
    return marker


__all__ = [
    "STAGE9_DIR", "STAGE9_PARTIAL", "ROOT_ENV", "STAGE9_MARKER", "BEGIN_FILE",
    "REQUIRED_OUTPUTS", "stage9_fingerprint", "stage9_root", "stage9_inputs",
    "tree_manifest", "require_stage9_complete",
]
