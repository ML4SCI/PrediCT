"""Shared configuration loader.

Reads ``configs/default.yaml`` and exposes typed constants used across stages.
This is the single source of truth for paths, seeds, thresholds, and cohort
exclusions.

Per-stage modules import from this module. Do not introduce a stage-local
``config.py``: extend this file or the YAML instead.

Decisions referencing this module:
    D004: Cohort exclusions at discovery
    D005: Target voxel grid
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo root: this file lives at src/predict/config.py
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    data_raw: Path
    data_xml: Path
    outputs: Path

    def __post_init__(self) -> None:
        # The DICOM loaders take data_raw.parent and append "raw/<pid>", so a
        # differently named data_raw would make them read another folder than
        # discovery did. Fail here instead of silently disagreeing.
        if self.data_raw.name != "raw":
            raise ValueError(
                f"paths.data_raw must be a folder named 'raw' (got {self.data_raw})")


@dataclass(frozen=True)
class Cohort:
    exclude_pids: tuple[str, ...]
    exclude_ge_scanners: bool


@dataclass(frozen=True)
class Resample:
    in_plane_mm: float
    slice_mm: float

    @property
    def target_spacing(self) -> tuple[float, float, float]:
        # SimpleITK convention: (x, y, z) in mm.
        return (self.in_plane_mm, self.in_plane_mm, self.slice_mm)


@dataclass(frozen=True)
class HU:
    agatston_threshold: int
    metal_artifact_threshold: int
    clip_min: int
    clip_max: int


@dataclass(frozen=True)
class LesionGrouping:
    max_inplane_dist_mm: float
    max_slice_gap: int


@dataclass(frozen=True)
class Stability:
    rotation_degrees: tuple[float, ...]
    translation_mm: tuple[float, ...]
    noise_sigma_hu: tuple[float, ...]
    background_fill_hu: float
    noise_seed_multiplier: int
    icc_threshold: float
    primary_gate: str = "classic"


@dataclass(frozen=True)
class Reduce:
    variance_threshold: float
    icc_threshold: float
    r2_linkage: str
    r2_elbow_min_gap: float
    r2_fallback_distance: float
    pca_cumvar: float


@dataclass(frozen=True)
class Config:
    paths: Paths
    seed: int
    cohort: Cohort
    resample: Resample
    hu: HU
    lesion_grouping: LesionGrouping
    stability: Stability
    reduce: Reduce
    raw: dict[str, Any]  # full parsed YAML, for stage-specific lookup


def _to_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML. Defaults to ``configs/default.yaml``."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        paths=Paths(
            data_raw=_to_path(raw["paths"]["data_raw"]),
            data_xml=_to_path(raw["paths"]["data_xml"]),
            outputs=_to_path(raw["paths"]["outputs"]),
        ),
        seed=int(raw["seed"]),
        cohort=Cohort(
            exclude_pids=tuple(str(p) for p in raw["cohort"]["exclude_pids"]),
            exclude_ge_scanners=bool(raw["cohort"]["exclude_ge_scanners"]),
        ),
        resample=Resample(
            in_plane_mm=float(raw["resample"]["in_plane_mm"]),
            slice_mm=float(raw["resample"]["slice_mm"]),
        ),
        hu=HU(
            agatston_threshold=int(raw["hu"]["agatston_threshold"]),
            metal_artifact_threshold=int(raw["hu"]["metal_artifact_threshold"]),
            clip_min=int(raw["hu"]["clip_min"]),
            clip_max=int(raw["hu"]["clip_max"]),
        ),
        lesion_grouping=LesionGrouping(
            max_inplane_dist_mm=float(raw["lesion_grouping"]["max_inplane_dist_mm"]),
            max_slice_gap=int(raw["lesion_grouping"]["max_slice_gap"]),
        ),
        stability=Stability(
            rotation_degrees=tuple(float(x) for x in raw["stability"]["rotation_degrees"]),
            translation_mm=tuple(float(x) for x in raw["stability"]["translation_mm"]),
            noise_sigma_hu=tuple(float(x) for x in raw["stability"]["noise_sigma_hu"]),
            background_fill_hu=float(raw["stability"]["background_fill_hu"]),
            noise_seed_multiplier=int(raw["stability"]["noise_seed_multiplier"]),
            icc_threshold=float(raw["stability"]["icc_threshold"]),
            primary_gate=str(raw["stability"].get("primary_gate", "classic")),
        ),
        reduce=Reduce(
            variance_threshold=float(raw["reduce"]["variance_threshold"]),
            icc_threshold=float(raw["reduce"]["icc_threshold"]),
            r2_linkage=str(raw["reduce"]["r2_linkage"]),
            r2_elbow_min_gap=float(raw["reduce"]["r2_elbow_min_gap"]),
            r2_fallback_distance=float(raw["reduce"]["r2_fallback_distance"]),
            pca_cumvar=float(raw["reduce"]["pca_cumvar"]),
        ),
        raw=raw,
    )


# Stability gate that feeds stages 6 onward, by name -> output subdirectory.
# "realistic" is the D033 composite perturbation gate (scripts/05b_icc_gate_realistic.py);
# "classic" is the D014 14-perturbation gate (scripts/05_icc_gate.py), also the
# layout of gate roots built by scripts/make_gate_root.py. A config without the
# key means "classic", so configs written before the key existed keep working.
GATE_DIRS: dict[str, str] = {"realistic": "05b_icc_realistic", "classic": "05_icc"}


# Written by stage 5b --allow-partial; such a folder must never feed stage 6.
PARTIAL_SMOKE_MARKER = "PARTIAL_SMOKE_DO_NOT_USE"


def gate_dir(cfg: Config) -> Path:
    """Directory holding icc_report.csv and gated_features.csv for the primary gate.

    The one place downstream stages resolve the gate, so it is also where a
    partial smoke gate is refused.
    """
    try:
        path = cfg.paths.outputs / GATE_DIRS[cfg.stability.primary_gate]
    except KeyError:
        raise ValueError(
            f"stability.primary_gate must be one of {sorted(GATE_DIRS)}, "
            f"got {cfg.stability.primary_gate!r}"
        ) from None
    if (path / PARTIAL_SMOKE_MARKER).exists():
        raise ValueError(f"{path} is a partial smoke gate ({PARTIAL_SMOKE_MARKER}); "
                         "rerun stage 5b on the full cohort")
    return path


@lru_cache(maxsize=1)
def default_config() -> Config:
    """Cached accessor for the default config. Tests can ignore this."""
    return load_config()


def default_n_jobs(cap: int | None = None) -> int:
    """Worker count for this run, resolved from the batch scheduler if present.

    Resolution order, first hit wins:

    1. ``PREDICT_N_JOBS`` : explicit override, always respected.
    2. ``PBS_NCPUS``      : set by some PBS Pro installs to what the job requested.
    3. ``NCPUS``          : what some PBS Pro clusters set instead of
       ``PBS_NCPUS``. Honoured only inside a
       PBS job (``PBS_JOBID`` set), so a stray ``NCPUS`` elsewhere is ignored.
    4. ``SLURM_CPUS_ON_NODE``: harmless to support, costs one line.
    5. A hardcoded **4**.

    ``PBS_NODEFILE`` is intentionally not consulted; see the comment in the
    body for why counting its lines is wrong on PBS Pro.

    Step 5 is deliberately NOT ``mp.cpu_count()``. Off the scheduler we are
    almost always on a shared login node, where ``cpu_count()`` reports the
    machine's full core count (64 on the current cluster) rather than the
    small share this session is entitled to. Spawning 64 workers there is
    antisocial and gets processes killed. A conservative 4 is the right
    answer for the only situations that reach this branch: a login-node
    smoke test, or a laptop.

    Never returns less than 1. ``cap`` clamps the result, which matters for
    the memory-heavy per-patient stages where one worker holds a full CT
    volume in RAM.

    Reading the scheduler rather than hardcoding means the job script is the
    single place parallelism is controlled. Requesting 12 cores and then
    spawning 80 workers is the classic way to make an HPC run slower than a
    laptop.
    """
    n: int | None = None

    explicit = os.environ.get("PREDICT_N_JOBS", "").strip()
    if explicit:
        try:
            n = int(explicit)
        except ValueError:
            n = None

    if n is None:
        pbs_ncpus = os.environ.get("PBS_NCPUS", "").strip()
        if pbs_ncpus:
            try:
                n = int(pbs_ncpus)
            except ValueError:
                n = None

    if n is None and os.environ.get("PBS_JOBID", "").strip():
        ncpus = os.environ.get("NCPUS", "").strip()
        if ncpus:
            try:
                n = int(ncpus)
            except ValueError:
                n = None

    # NOTE: there is deliberately NO ``PBS_NODEFILE`` line-count branch here.
    # That idiom is correct on Torque / OpenPBS, where the nodefile holds one
    # line per allocated core. **PBS Pro writes one line per chunk**, so under
    # ``select=1:ncpus=12`` the file has a single line and counting it yields
    # 1, not 12. That is strictly worse than the default below, and it would
    # bite hardest on stages 2, 3 and 4, which run_pipeline.sh invokes with no
    # worker flag at all and which therefore depend on this function outright.
    # If you are tempted to re-add it, check ``wc -l < $PBS_NODEFILE`` inside a
    # real job on this cluster first.

    if n is None:
        slurm = os.environ.get("SLURM_CPUS_ON_NODE", "").strip()
        if slurm:
            try:
                n = int(slurm)
            except ValueError:
                n = None

    if n is None:
        n = 4  # off-scheduler fallback; never cpu_count() (login node reports 64)

    n = max(1, int(n))
    if cap is not None:
        n = min(n, max(1, int(cap)))
    return n


# Canonical vessel names. The four full-name strings COCA emits map to these.
# Dirty / unknown names are returned as ``None`` from the parser.
VESSEL_NAME_MAP: dict[str, str] = {
    "Left Anterior Descending Artery": "LAD",
    "Right Coronary Artery": "RCA",
    "Left Circumflex Artery": "LCx",
    "Left Coronary Artery": "LM",
}
VESSEL_NAMES: tuple[str, ...] = ("LAD", "RCA", "LCx", "LM")
