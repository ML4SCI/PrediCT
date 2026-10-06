# Reproducibility

## Determinism
- All randomised steps read `seed` from `configs/default.yaml` (default: 42)
  or a seed fixed in the decision that defines them; seeds are passed
  explicitly.
- Parallel bootstraps derive a per-iteration seed, so results do not depend
  on the number of workers.
- PyRadiomics is deterministic given the same `params.yaml` and input.

## Environment
- Python 3.8; `pyproject.toml` declares the package, `requirements.txt` the
  dependencies.
- Install: `pip install -e .` from the repo root (PyRadiomics from conda).
- Every stage writes a run header with library versions and input hashes.

## Stage outputs and completion records
- Every stage writes CSV / NPY / JSON at its boundary under
  `outputs/<NN_stage>/`.
- Each stage publishes its folder in one step with a completion record
  (input hashes, code fingerprint, output hashes); consumers refuse a
  missing, stale or partial result.
- Rerunning a stage requires only the published outputs of the stages it
  reads (`run_pipeline.sh --from N --to M`).
- The synced code revision is recorded in each record (`REVISION` file
  written at sync time, since the run machine may not hold the git history).

## Logs
- `run_pipeline.sh` writes one log per stage and a master log to
  `outputs/_logs/`.

## Decision tracking
- One markdown per decision in `decisions/D###-slug.md`, dated; rules and
  interpretations are fixed before the results they govern.
- Code that implements a decision cites it in the docstring header.
