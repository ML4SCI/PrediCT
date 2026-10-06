# D043 Stage 9 integrity: one published chain, termination-aware results, consistent balance, stratum-level replication, complete lesion stability

**Date**: 2026-10-05
**Stage**: exploratory lesion chain (stage 9)
**Status**: Active (amends the LAD-phenotype, LM-isolated and lesion-stability implementations; every pre-registered threshold and hypothesis is unchanged)
**Module**: `src/predict/stage9.py`, `scripts/09_stage9.py`, `run_pipeline.sh`, `experiments/lad_phenotype/{run,matched,finalise,propensity}.py`, `experiments/lm_isolated_low_burden/{run,finalise}.py`, `experiments/lesion_morphology/lesion_stability_d033.py`, `src/predict/discover/cluster_discovery.py`

## Decision

1. **One published chain.** All stage 9 scripts read and write `stage9_root(cfg)`. `run_pipeline.sh` runs `scripts/09_stage9.py begin` (inputs verified: stage 7 record, lesions.csv, xml_roundtrip.csv, and the calibration stage 4b recorded; empty `outputs/exploratory.partial/`), the chain with `PREDICT_STAGE9_ROOT=exploratory.partial`, and `finish` (inputs and code unchanged, required outputs present, LAD termination records consistent), which writes `COMPLETE_stage9.json` and publishes the folder as a whole. The published folder is read-only to the chain scripts; the figures call `require_stage9_complete`.
2. **Terminations are results.** `matched.py` deletes every earlier matched output first and records `no_signature`, `no_carriers` or `match_infeasible` as the verdict; `run.py` deletes earlier step outputs. The LAD finaliser reads a matched comparison only where the verdict is `match_ok` and stops on any other combination.
3. **Balance on the compared population.** The LAD post-match SMD is computed on the matched cases (each once) against the matched controls (each once), the two groups the outcome tests compare. The former pair-weighted value is kept as `smd_post_pair_weighted`. The pre-registered threshold (< 0.1) is unchanged.
4. **Stratum replication is computed in the stratum.** A stratum run keeps the shared lesion partition but recomputes the per-cluster medians and vessel obs/exp on the stratum's own lesions before applying the P1 criteria; the full-cohort run checks that the same functions reproduce the stored tables. Replication is reported per component (signature, axial, matched) as replicates / does_not_replicate / not_evaluable; `overall_replicates` is True only when every component replicates and is never True without results. Each stratum is judged against the full cohort's verdict, not its own pass (plan P5: "each verdict must hold"): the axial component replicates when the stratum reaches the full verdict or its effect is within +/-0.10 of the full effect; the matched component replicates when the stratum reaches the full verdict (its summary has no single effect estimate, so the tolerance alternative does not apply). A reversed verdict is therefore not replicated, and a preserved negative verdict is.
5. **No historical LAD reference.** The LM analysis uses this run's LAD signature only. A missing or malformed signature stops it; an empty signature makes P5 not evaluable (and the overall LM verdict None).
6. **Complete lesion stability.** Lesions created by regrouping are kept and count against the C8 Jaccard together with lost lesions. Before any verdict the baseline must equal lesions.csv on every key and every clustered input, the replayed z-scoring must reproduce lesion_features.csv, and the production k-means (`fit_kmeans`, same k and seed) must reproduce the stored labels exactly; perturbed lesions are assigned with that model. Workers use the given config; the calibration is the one stage 4b recorded.

## Rationale

A terminated match left the previous "LAD-bias survives burden matching" summary to be read alongside `match_infeasible`. The SMD repeated each case once per control while the outcome tests counted it once (reported 0.000; compared population 0.408). Stratum runs reused the full-cohort tables, so the stratum signature could only reproduce the full one, and the replication finaliser returned True on no results. The LM analysis substituted clusters 10 and 11 for a missing or empty signature. The stability worker dropped regrouped lesions (a split C8 lesion scored 1.0 instead of 0.5), printed parity without enforcing it, and ignored `--config`.

## Consequence

`fit_cluster`'s k-means path now goes through `fit_kmeans` (identical results), which changes the stage 6 to 8 code fingerprints: stages 6 to 9 are recomputed. Every reported stage 9 number is computed under these rules.

Fingerprint scope amended by D044: stage 9 hashes `figures/__init__.py` and `figures/_lesion_classes.py`, not the whole figures package.

## Verified by

- `tests/experiments/test_stage9_chain.py`: root guards; begin / finish publication; finish refuses an incomplete chain and a stale comparison; the record refuses added files; the LAD finaliser refuses a stale comparison and never reports replication without results; strata are judged against the full verdict in both directions, and the axial tolerance still applies; balance on the compared population (0.408 vs 0.000); the C8 Jaccard counts additions (0.5); baseline parity enforced on every clustered input.
- `experiments/lm_isolated_low_burden/tests/test_run.py`: no fallback; P5 not evaluable without LAD clusters; missing or malformed signatures stop.
