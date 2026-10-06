"""Stage 8 D031: cross-cohort ARI consolidation.

Pure re-exporter of ``outputs/07_analyse/cross_cohort_ari.csv`` into
``outputs/08_validate/cross_cohort_ari_consolidated.csv`` with explicit
PASS columns at the D027 threshold (0.80).

No recomputation. No new statistics. This module exists so that stage 8's
output directory is self-contained for the paper's Methods.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# D027 / D031 partition-stability bar
ARI_PASS_THRESHOLD: float = 0.80

# Source columns expected on the stage-7 file.
SOURCE_REQUIRED_COLS: tuple[str, ...] = (
    "partition",
    "stratum",
    "n_shared_pids",
    "ari",
    "passes",
)


def consolidate(
    stage7_path: Path,
    pass_threshold: float = ARI_PASS_THRESHOLD,
) -> pd.DataFrame:
    """Read stage 7 cross_cohort_ari.csv and append PASS verdict columns.

    Output columns:
      partition, stratum, n_shared_pids, ari, pass_threshold, pass_verdict
    """
    if not stage7_path.exists():
        raise FileNotFoundError(
            f"D031 cross_cohort_ari requires stage 7 output at "
            f"{stage7_path}. Run scripts/08_analyse.py first."
        )
    df = pd.read_csv(stage7_path)
    missing = set(SOURCE_REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{stage7_path} missing expected stage-7 columns: {sorted(missing)}"
        )
    out = df[["partition", "stratum", "n_shared_pids", "ari"]].copy()
    out["pass_threshold"] = float(pass_threshold)
    out["pass_verdict"] = (df["ari"] >= pass_threshold).astype(bool)
    return out


def write(
    stage7_path: Path,
    out_path: Path,
    pass_threshold: float = ARI_PASS_THRESHOLD,
) -> pd.DataFrame:
    """Convenience: consolidate + write to CSV. Returns the dataframe."""
    df = consolidate(stage7_path, pass_threshold=pass_threshold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
