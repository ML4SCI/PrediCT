"""Stage 7 helper: re-derive the two D019 derived features on RAW values.

Background:

  Stage 5's ``prepare_matrix.py`` creates two derived features
  (``high_density_fraction`` and ``vessel_burden_gini``) from the canonical
  per-vessel inputs. Those derived columns live in
  ``outputs/06_reduce/prepared_matrix.csv`` AFTER D019 z-scoring; they are
  NOT written to the raw ``outputs/03_features/features.csv``.

  Stage 7 reads RAW values from features.csv (per D023's clinical-
  interpretability rule). If the derived columns are needed at raw scale
  (e.g., for the D025 directional hypothesis on ``vessel_burden_gini``),
  this helper re-derives them in-place with the stage-5 functions.

Definitions (the stage-5 functions, D019 / D040):

* ``high_density_fraction = (d3 + d4 count) / (d1 + d2 + d3 + d4 count)``,
  summed across all 4 vessels. Patients with zero total tier-bin count
  return 0.0 (no division by zero).

* ``vessel_burden_gini`` = Gini coefficient across the 4 per-vessel
  Agatston values, calcium-free vessels included (D040): 0 for burden spread
  evenly, 0.75 for all burden in one vessel.

Both are the stage-5 functions themselves (imported, not re-implemented), so
stage 5 and stage 7 cannot drift apart.
"""
from __future__ import annotations

import pandas as pd

from predict.reduce.prepare_matrix import (
    compute_high_density_fraction as derive_high_density_fraction,
    compute_vessel_burden_gini as derive_vessel_burden_gini,
)

VESSELS: tuple[str, ...] = ("lad", "rca", "lcx", "lm")


def augment_raw_with_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``high_density_fraction`` and
    ``vessel_burden_gini`` appended (if their inputs are available).

    Existing columns of those names are PRESERVED (we do not overwrite).
    This is defensive in case a future stage 3 starts writing them.
    """
    out = df.copy()
    if "high_density_fraction" not in out.columns:
        derived = derive_high_density_fraction(out)
        if derived is not None:
            out["high_density_fraction"] = derived
    if "vessel_burden_gini" not in out.columns:
        derived = derive_vessel_burden_gini(out)
        if derived is not None:
            out["vessel_burden_gini"] = derived
    return out
