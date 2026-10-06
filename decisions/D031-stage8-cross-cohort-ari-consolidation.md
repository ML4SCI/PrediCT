# D031 Stage 8 cross-cohort ARI consolidation

**Date**: 2026-06-08
**Stage**: validate (stage 8)
**Status**: Active
**Module**: `src/predict/validate/cross_cohort_ari.py`, `scripts/09_validate.py`

## Decision

Stage 7 already produces `outputs/07_analyse/cross_cohort_ari.csv`
(per D027 verification), which contains the Adjusted Rand Index between
the full-cohort and each kernel-stratified subcohort for both the
spatial-only k=2 and the forced k=3 partitions on the patients shared
between cohorts.

Stage 8 re-exports this file to
`outputs/08_validate/cross_cohort_ari_consolidated.csv` with two
additional bookkeeping columns:

| Column | Source | Description |
|---|---|---|
| existing columns | re-exported as-is | per-cohort-pair, per-partition ARI + n_shared_pids |
| `pass_threshold` | constant 0.80 | partition-level stability bar (D027 inherited) |
| `pass_verdict` | computed | `True` iff ARI ≥ 0.80 |

No recomputation. No new statistics. The point of D031 is to give
stage 8 a single deliverable for the validation evidence so that the
paper Methods section can cite ONE file as "the cross-cohort
phenotype-replication audit" rather than telling the reader to
cross-reference a stage 7 file.

## Rationale

### Why re-export instead of read-through

- Stage 8 outputs need to be self-contained so that a reader
  inspecting `outputs/08_validate/` can verify the full validation
  story without bouncing into stage 7.
- The `pass_threshold` + `pass_verdict` columns add a verdict layer
  that does not belong in stage 7's raw ARI output (stage 7 reports
  values; stage 8 reports verdicts).

### Why 0.80, not the D030 simulated threshold T

The cross-cohort ARI is computed on shared pids between cohorts
(typically N ≈ 200) and compares two FULL fits on different cohorts
(not a train-vs-held-out comparison). The 0.80 threshold inherits
from D027 which uses it as the cross-cohort partition-stability bar.

D030's simulated T is for per-fold LOO predictions on N ≈ 42; the
sampling-variance considerations are different, so D031 keeps the
0.80 convention.

### Why this is its own decision doc

- Keeps the stage-8 deliverables triad (D029 + D030 + D031)
  symmetrical: one decision per output file.
- Pins the 0.80 threshold to a written rationale rather than letting
  it float as a magic number in the re-exporter.

## Output schema

`outputs/08_validate/cross_cohort_ari_consolidated.csv` (re-exported
from `outputs/07_analyse/cross_cohort_ari.csv`):

| Column | Type | Description |
|---|---|---|
| `partition` | str | `spatial_k2` or `burden_k3` |
| `stratum` | str | stratified cohort label, e.g. `Qr36d_2`, `I30f_3` (full cohort is implicit reference) |
| `n_shared_pids` | int | pids present in both full and stratum |
| `ari` | float | from stage 7 (`partition_ari_table`) |
| `pass_threshold` | float | 0.80 (constant, D027 inherited) |
| `pass_verdict` | bool | `ari >= 0.80` |

Source columns required on the stage-7 file:
`{partition, stratum, n_shared_pids, ari, passes}` (the stage 7 `passes`
column already encodes the verdict at 0.80; D031 renames it to
`pass_verdict` and adds the explicit `pass_threshold` for self-documentation).

Expected row count: 2 partitions × 2 strata = 4 rows for the default
production sweep (spatial_k2 × {Qr36d_2, I30f_3} + burden_k3 ×
{Qr36d_2, I30f_3}). Optional extra rows if the robust-low-burden
cohort is added to stage 7's strata.

## Out of scope

- Recomputing ARI (stage 7 already does this).
- Anything beyond bookkeeping; the file is a pure re-exporter.

## Verified by

- `tests/validate/test_cross_cohort_ari.py`:
  - re-exported file has all rows from the stage 7 source
  - `pass_threshold` column is exactly 0.80 on every row
  - `pass_verdict` matches `ari >= 0.80` on every row
  - re-running the export twice produces byte-identical output
- `tests/validate/test_orchestrator.py`:
  - `cross_cohort_ari_consolidated.csv` is produced when stage 7
    outputs are present; raises if stage 7 has not been run
