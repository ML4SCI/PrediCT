"""Input fingerprints and completion markers for resumable stages.

Stage 4 (D014 perturbations, D033 realistic draws) takes hours and resumes
from per-patient / per-perturbation files. A file may be reused only when it
was produced from exactly the current inputs and code, and finished without
errors; otherwise an upstream rerun could be silently mixed with old
results. A stage that finishes cleanly
writes ``COMPLETE.json``; the gate that consumes it refuses to run unless
the marker exists and matches the current inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from predict.config import REPO_ROOT

MARKER = "COMPLETE.json"

# What stage 1-4 outputs depend on: the packages of stages 1-4 and the config
# sections they read. Stage 5+ code (reduce, discover, analyse, validate,
# figures) and config sections (reduce, consensus, gap_statistic,
# stability_cohort, hopkins) cannot change a perturbation draw, so editing
# them must not invalidate hours of stage 4 results.
UPSTREAM_PACKAGES = ("io", "preprocess", "features", "stability")
# Top-level modules that stage 1-4 COMPUTATION imports. Bookkeeping modules
# (completion: fingerprints; provenance: run headers; publish: folder moves)
# change no computed value, so editing them must not invalidate results.
# The first version hashed every top-level module, and the atomic-publish fix
# in publish.py then made stage 5b refuse valid stage 4 draws.
UPSTREAM_TOP_LEVEL = ("__init__.py", "config.py")
UPSTREAM_CONFIG_SECTIONS = ("seed", "cohort", "resample", "hu", "lesion_grouping", "stability")


def file_sha(path: Path) -> str:
    """SHA-256 of a file's bytes, or ``"missing"``."""
    path = Path(path)
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha(extra: Iterable[Path] = (), repo_root: Path = REPO_ROOT,
             packages: Iterable[str] | None = None) -> str:
    """One hash over ``params.yaml``, any ``extra`` files (the stage
    script), and either every ``src/predict`` module (``packages`` None) or
    the ``UPSTREAM_TOP_LEVEL`` modules plus the modules of ``packages``."""
    root = Path(repo_root)
    pkg_root = root / "src" / "predict"
    if packages is None:
        files = sorted(pkg_root.rglob("*.py"))
    else:
        files = [pkg_root / m for m in UPSTREAM_TOP_LEVEL]
        for pkg in packages:
            files += sorted((pkg_root / pkg).rglob("*.py"))
    files += [root / "params.yaml", *map(Path, extra)]
    h = hashlib.sha256()
    for p in files:
        h.update(str(Path(p).resolve().relative_to(root.resolve())).replace("\\", "/").encode())
        h.update(file_sha(p).encode())
    return h.hexdigest()


def upstream_fingerprint(cfg: Any, extra: Iterable[Path] = ()) -> str:
    """Code and configuration that stage 1-4 results depend on: the stage 1-4
    packages, ``UPSTREAM_TOP_LEVEL``, params.yaml, the given stage script(s),
    and the config sections in ``UPSTREAM_CONFIG_SECTIONS``. Recorded by a
    resumable stage and recomputed by the gate that consumes it, through
    this one function."""
    return digest({
        "code": code_sha(extra=extra, packages=UPSTREAM_PACKAGES),
        "config": {k: cfg.raw.get(k) for k in UPSTREAM_CONFIG_SECTIONS},
    })


def digest(payload: Any) -> str:
    """Stable SHA-256 of a JSON-serialisable payload."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def clear_marker(stage_dir: Path, name: str = MARKER) -> None:
    (Path(stage_dir) / name).unlink(missing_ok=True)


def write_marker(stage_dir: Path, payload: dict, name: str = MARKER) -> None:
    write_json_atomic(Path(stage_dir) / name, payload)


def files_manifest(folder: Path, names: Iterable[str]) -> dict[str, str]:
    """{name: sha256} for the given files in ``folder``."""
    return {n: file_sha(Path(folder) / n) for n in sorted(names)}


def require_marker(stage_dir: Path, expect: dict, name: str = MARKER) -> dict:
    """Return the stage's marker, or stop if it is missing or any ``expect``
    key differs (e.g. features.csv changed after the stage ran)."""
    path = Path(stage_dir) / name
    if not path.exists():
        raise SystemExit(f"hard stop: {path} missing; the stage did not finish cleanly "
                         "(errors, failed parity, smoke run, or never run)")
    marker = json.loads(path.read_text(encoding="utf-8"))
    bad = [k for k, v in expect.items() if marker.get(k) != v]
    if bad:
        raise SystemExit(f"hard stop: {path} does not match the current inputs ({', '.join(bad)}); "
                         "rerun the stage")
    return marker
