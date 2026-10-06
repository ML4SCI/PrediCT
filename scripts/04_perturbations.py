#!/usr/bin/env python
"""Stage 4a, perturbation re-extraction.

For each eligible patient and each of the 14 perturbations (D014), apply the
perturbation to the preprocessed CT (mask held fixed per D014),
re-extract the 107 PyRadiomics features, and write one CSV per perturbation.

Per D015, eligible patients are those with ``radiomics_status == "ok"`` in
``outputs/03_features/features.csv`` (444 minus 22 small-mask skips = 422).

Per D016, the 68 canonical features are invariant by construction and are NOT
re-extracted here. Only the 107 ``original_*`` PyRadiomics columns are
recomputed.

Outputs (under ``outputs/04_perturbations/``):

  - ``{perturbation_name}.csv``     one row per patient, columns
                                    ``pid`` + 107 ``original_*`` + ``error``.
  - ``run_header.json``             reproducibility breadcrumbs.
  - ``perturbation_log.csv``        per-(pid, pert) runtime + status row.

The script is **resumable**: a per-perturbation CSV is skipped on rerun only
if its sidecar ``{perturbation_name}.json`` carries the current fingerprint
(all src/predict code, params.yaml, configs/default.yaml, this script, the
perturbation spec, the eligible pid list and the published stage-2 folder)
and recorded zero errors, and the CSV holds exactly the eligible pids with an
empty ``error`` column.

Any patient error (including an empty mask, which stage 2 rules out) makes
the script exit 1 without ``COMPLETE.json``; stage 5 (05_icc_gate.py) refuses
to gate without a ``COMPLETE.json`` that matches the current features.csv.
``--limit`` and ``--only-perturbation`` never write it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from predict.completion import (
    clear_marker, digest, file_sha, upstream_fingerprint, write_json_atomic, write_marker,
)
from predict.provenance import code_revision
from predict.config import REPO_ROOT, Config, default_n_jobs, load_config
from predict.features.radiomics import create_extractor, extract_pyradiomics
from predict.io.spacing import load_spacing_metadata
from predict.stability.perturbations import (
    PerturbationSpec,
    apply_perturbation,
    enumerate_perturbations,
)


# ───────────────────── reproducibility breadcrumbs ─────────────────────


def _file_sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def build_run_header(
    repo_root: Path,
    params_yaml: Path,
    config_yaml: Path,
    specs: tuple[PerturbationSpec, ...],
) -> dict:
    import sys as _sys
    info = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": code_revision(repo_root),
        "params_yaml_sha": _file_sha(params_yaml),
        "config_yaml_sha": _file_sha(config_yaml),
        "python_version": _sys.version.split()[0],
        "perturbation_count": len(specs),
        "perturbation_names": [s.name for s in specs],
    }
    for mod in ("numpy", "pandas", "SimpleITK", "radiomics"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            info[f"{mod}_version"] = "n/a"
    return info


# ───────────────────── per-(pid, perturbation) worker ─────────────────────


# Worker-process scope: cached extractor and config to amortise init.
_worker_state: dict = {}


def _init_worker(config_path_str: str | None) -> None:
    cfg = load_config(Path(config_path_str) if config_path_str else None)
    _worker_state["cfg"] = cfg
    _worker_state["extractor"] = create_extractor()
    preproc = cfg.paths.outputs / "02_preprocessed"
    _worker_state["spacing"] = load_spacing_metadata(preproc / "spacing.json")
    _worker_state["preproc"] = preproc


def _process_one(
    pid: str,
    pert: PerturbationSpec,
) -> dict:
    t0 = time.perf_counter()
    cfg: Config = _worker_state["cfg"]
    spacing = _worker_state["spacing"]
    preproc = _worker_state["preproc"]
    extractor = _worker_state["extractor"]

    try:
        # Load preprocessed CT and mask.
        ct_np = np.load(preproc / f"{pid}_ct.npy")
        mask_np = np.load(preproc / f"{pid}_mask.npy")
        ct_img = sitk.GetImageFromArray(ct_np.astype(np.float32))
        ct_img.SetSpacing(spacing)

        # Apply perturbation (mask not touched, D014).
        perturbed_ct = apply_perturbation(ct_img, pert, pid=pid, cfg=cfg)
        perturbed_arr = sitk.GetArrayFromImage(perturbed_ct).astype(np.int16)

        if mask_np.sum() == 0:
            # Stage 2 guarantees a non-empty mask, so this is an error.
            return {
                "pid": pid,
                "perturbation": pert.name,
                "status": "error",
                "error": "mask is empty",
                "runtime_sec": round(time.perf_counter() - t0, 2),
                "features": {},
            }

        features = extract_pyradiomics(
            perturbed_arr, mask_np, spacing, extractor, pid=f"{pid}:{pert.name}",
        )
        return {
            "pid": pid,
            "perturbation": pert.name,
            "status": "ok",
            "error": "",
            "runtime_sec": round(time.perf_counter() - t0, 2),
            "features": features,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "pid": pid,
            "perturbation": pert.name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_sec": round(time.perf_counter() - t0, 2),
            "features": {},
        }


def _worker_tuple(args) -> dict:
    pid, pert = args
    return _process_one(pid, pert)


# ───────────────────── eligibility and resume helpers ─────────────────────


def _eligible_pids(features_csv: Path) -> list[str]:
    """D015: 422 patients with radiomics_status == 'ok'."""
    df = pd.read_csv(features_csv, dtype={"pid": str})
    if "radiomics_status" not in df.columns:
        raise KeyError(
            f"{features_csv} missing 'radiomics_status' column; rerun stage 3."
        )
    eligible = df.loc[df["radiomics_status"] == "ok", "pid"].astype(str).tolist()
    return eligible


def _stage2_stamp(preproc: Path, pids: list[str]) -> dict:
    """Identity of the published stage-2 folder: its reports by content and
    each eligible patient's arrays by size and modification time (stage 2
    republishes the whole folder on every run, so a rerun changes them)."""
    def stamp(p: Path) -> str:
        st = p.stat() if p.exists() else None
        return f"{st.st_size}:{st.st_mtime_ns}" if st else "missing"
    return {
        "reports": [file_sha(preproc / n) for n in
                    ("spacing.json", "preprocess_report.csv", "xml_roundtrip.csv")],
        "arrays": {p: [stamp(preproc / f"{p}_ct.npy"), stamp(preproc / f"{p}_mask.npy")]
                   for p in pids},
    }


def run_fingerprint(cfg: Config, pids: list[str]) -> str:
    """Stage 4a run fingerprint: stage 1-4 code and config (including this
    script), the eligible pid list and the published stage-2 folder.
    Recorded in COMPLETE.json; stage 5 recomputes it with this function."""
    return digest({
        "upstream": upstream_fingerprint(cfg, extra=[REPO_ROOT / "scripts" / "04_perturbations.py"]),
        "pids": list(pids),
        "stage2": _stage2_stamp(cfg.paths.outputs / "02_preprocessed", list(pids)),
    })


def _csv_is_complete(csv_path: Path, expected_pids: list[str], fingerprint: str) -> bool:
    """True if the CSV may be reused: matching sidecar fingerprint, zero
    recorded errors, exactly the expected pids once each, no error text."""
    meta_path = csv_path.with_suffix(".json")
    if not (csv_path.exists() and meta_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        existing = pd.read_csv(csv_path, usecols=["pid", "error"], dtype={"pid": str},
                               keep_default_na=False)
    except Exception:  # noqa: BLE001  (unreadable = recompute)
        return False
    if meta.get("fingerprint") != fingerprint or meta.get("n_error") != 0:
        return False
    pids = existing["pid"].astype(str)
    if pids.duplicated().any() or set(pids) != set(expected_pids):
        return False
    return bool((existing["error"].astype(str).str.strip() == "").all())


# ───────────────────── main ─────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-workers", type=int, default=default_n_jobs(cap=16),
                        help="Worker processes. Defaults to the scheduler allocation "
                             "(PBS_NCPUS / PBS_NODEFILE), capped at 16 because each "
                             "worker holds a full CT volume in RAM.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap eligible patient count (for smoke tests).")
    parser.add_argument("--only-perturbation", type=str, default=None,
                        help="Process a single perturbation by name (e.g. 'noise_5').")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force re-extraction even if a per-perturbation CSV "
                             "is already complete.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("stability")

    cfg: Config = load_config(args.config)
    out_dir = cfg.paths.outputs / "04_perturbations"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_marker(out_dir)   # set again only if this run finishes cleanly
    features_csv = cfg.paths.outputs / "03_features" / "features.csv"

    # Eligible cohort (D015).
    pids = _eligible_pids(features_csv)
    if args.limit:
        pids = pids[: args.limit]
    log.info("Stage 4a: eligible patients = %d (D015)", len(pids))

    # Perturbation set (D014).
    specs = enumerate_perturbations(cfg)
    if args.only_perturbation is not None:
        sel = [s for s in specs if s.name == args.only_perturbation]
        if not sel:
            valid = ", ".join(s.name for s in specs)
            raise SystemExit(
                f"unknown perturbation {args.only_perturbation!r}. Valid: {valid}"
            )
        specs = tuple(sel)
    log.info("Perturbations to run: %d (%s)", len(specs),
             ", ".join(s.name for s in specs))

    # Run header.
    header = build_run_header(
        repo_root=cfg.paths.outputs.parent,
        params_yaml=cfg.paths.outputs.parent / "params.yaml",
        config_yaml=args.config or (cfg.paths.outputs.parent / "configs" / "default.yaml"),
        specs=specs,
    )
    (out_dir / "run_header.json").write_text(
        json.dumps(header, indent=2) + "\n", encoding="utf-8",
    )

    # Aggregate per-(pid, pert) log across all perturbations in this run.
    all_logs: list[dict] = []
    run_fp = run_fingerprint(cfg, pids)
    total_errors = 0

    for spec in specs:
        out_csv = out_dir / f"{spec.name}.csv"
        spec_fp = digest({"run": run_fp, "spec": repr(spec)})
        if not args.no_resume and _csv_is_complete(out_csv, pids, spec_fp):
            log.info("[SKIP] %s: verified complete CSV already on disk", spec.name)
            continue

        tasks = [(pid, spec) for pid in pids]
        rows: list[dict] = []

        if args.n_workers == 1:
            _init_worker(str(args.config) if args.config else None)
            for t in tqdm(tasks, desc=spec.name):
                rows.append(_worker_tuple(t))
        else:
            with ProcessPoolExecutor(
                max_workers=args.n_workers,
                initializer=_init_worker,
                initargs=(str(args.config) if args.config else None,),
            ) as pool:
                futures = {pool.submit(_worker_tuple, t): t[0] for t in tasks}
                for fut in tqdm(as_completed(futures),
                                total=len(futures), desc=spec.name):
                    rows.append(fut.result())

        # Sort by pid for stable CSV order.
        rows.sort(key=lambda r: int(r["pid"]) if r["pid"].isdigit() else r["pid"])

        # Build per-perturbation dataframe: pid + 107 original_* + error.
        records: list[dict] = []
        for r in rows:
            rec = {"pid": r["pid"], "error": r["error"]}
            rec.update(r["features"])
            records.append(rec)
            all_logs.append({
                "pid": r["pid"],
                "perturbation": r["perturbation"],
                "status": r["status"],
                "runtime_sec": r["runtime_sec"],
                "error": r["error"],
                "n_features": len(r["features"]),
            })

        df = pd.DataFrame(records)
        # Ensure pid is first, error is last.
        cols = ["pid"] + [c for c in df.columns if c not in ("pid", "error")] + ["error"]
        df = df[cols]
        tmp = out_csv.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(out_csv)

        ok = sum(1 for r in rows if r["status"] == "ok")
        errors = sum(1 for r in rows if r["status"] != "ok")
        total_errors += errors
        # Sidecar last: its presence means the CSV above is complete.
        write_json_atomic(out_csv.with_suffix(".json"), {
            "fingerprint": spec_fp, "n_error": errors, "n_ok": ok,
            "revision": code_revision(cfg.paths.outputs.parent),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        log.info("[DONE] %s -> %s | ok=%d errors=%d", spec.name, out_csv.name, ok, errors)

    # Append per-(pid, pert) log.
    if all_logs:
        log_df = pd.DataFrame(all_logs)
        log_path = out_dir / "perturbation_log.csv"
        if log_path.exists():
            existing = pd.read_csv(log_path, dtype={"pid": str})
            log_df = pd.concat([existing, log_df], ignore_index=True)
        log_df.to_csv(log_path, index=False)

    if total_errors:
        log.error("Stage 4a FAILED: %d patient-perturbation errors (see perturbation_log.csv). "
                  "No COMPLETE.json; stage 5 will refuse to gate.", total_errors)
        return 1
    if args.limit or args.only_perturbation:
        log.warning("partial run (--limit / --only-perturbation): no COMPLETE.json written.")
        return 0
    write_marker(out_dir, {
        "features_csv_sha": file_sha(features_csv),
        "run_fingerprint": run_fp,
        "perturbations": [s.name for s in specs],
        "n_patients": len(pids),
        "revision": code_revision(cfg.paths.outputs.parent),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    log.info("Stage 4a complete. Outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
