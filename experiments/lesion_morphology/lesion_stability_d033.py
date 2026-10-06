#!/usr/bin/env python
"""Lesion-level stability under the D033 annotation perturbation.

The D033 gate tested patient-level features only. Lesion morphology
(volume, area, n_rois) is computed from the expert polygons, and lesion HU
(mean_hu_weighted, max_hu) is read from the XML, so of the D033 factors only
M1 (polygon boundary jitter) can change a lesion. This replays the 20 D033
primary draws' M1 jitter (same seeds, production code: sample_draw,
jitter_parse_result, compute_canonical_features) and re-groups lesions, then
measures:

  1. parity: unperturbed lesions equal 03_features/lesions.csv
  2. grouping: per draw, the fraction of baseline lesions matched to a
     re-grouped lesion (same vessel, Hungarian on centroid distance,
     gate MATCH_MM)
  3. ICC(1,1) per clustered morphology feature over the 20 draws, on lesions
     matched in every draw (the run.py D037 transform: log1p volume, mean HU,
     max HU, log1p n_rois)
  4. cluster stability: the stored k-means partition is rebuilt with the
     production k-means (``fit_kmeans``, same k and seed) and must
     reproduce the stored labels exactly; each draw lesion is z-scored with
     the baseline means / SDs and assigned with that model; per draw, the
     fraction of matched lesions keeping their cluster and the Jaccard of
     the C8-like set. Lost lesions AND lesions created by regrouping count
     against the Jaccard (dropping unmatched lesions would let a split C8
     lesion score 1.0; D043).

Integrity (D043): the baseline must equal
03_features/lesions.csv on every key and every clustered input (volume,
area, mean and max HU, ROI count) and reproduce the stored z-scored
features, before any verdict; workers use the given --config; the
calibration is the one stage 4b recorded.

Rules fixed before running: lesion morphology is STABLE under D033 iff every
clustered feature has ICC(1,1) >= 0.75 (the D033 threshold); C8 membership is
STABLE iff the median per-draw C8 Jaccard >= 0.75. HU comes from the XML
ROI values: mean_hu_weighted moves with the area weights, and both HU
features move when M1 regroups ROIs into different lesions. Neither is
claimed as robustness to inter-scan HU variability, which is outside D033.

Usage:
    python experiments/lesion_morphology/lesion_stability_d033.py [--config CONFIG]
Outputs: <outputs>/exploratory/lesion_morphology/lesion_stability_d033/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import json

from predict.completion import file_sha
from predict.config import default_n_jobs, load_config
from predict.discover.cluster_discovery import fit_kmeans
from predict.stage9 import stage9_root

KEYS = ["pid", "vessel", "lesion_idx"]
FEATURES = ("log1p_volume", "mean_hu_weighted", "max_hu", "log1p_n_rois")   # D037
N_DRAWS = 20
MATCH_MM = 3.0
ICC_MIN, C8_JACCARD_MIN = 0.75, 0.75


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """run.py's morphology transform (D037 four features), before z-scoring."""
    out = pd.DataFrame(index=df.index)
    out["log1p_volume"] = np.log1p(df["volume_mm3"])
    out["mean_hu_weighted"] = df["mean_hu_weighted"]
    out["max_hu"] = df["max_hu"]
    out["log1p_n_rois"] = np.log1p(df["n_rois"])
    return out


def _flatten(pid: str, lesions_per_vessel) -> list[dict]:
    return [{"pid": pid, "vessel": v, "lesion_idx": i,
             "cx": l.centroid_mm[0], "cy": l.centroid_mm[1], "cz": l.centroid_mm[2],
             "n_rois": l.n_rois, "total_area_mm2": l.total_area_mm2, "volume_mm3": l.volume_mm3,
             "mean_hu_weighted": l.mean_hu_weighted, "max_hu": l.max_hu}
            for v, ls in lesions_per_vessel.items() for i, l in enumerate(ls)]


def _match(base: pd.DataFrame, draw: pd.DataFrame) -> dict[int, int]:
    """baseline row index -> draw row index, per vessel, Hungarian on centroids."""
    from scipy.optimize import linear_sum_assignment
    out: dict[int, int] = {}
    for v, b in base.groupby("vessel"):
        d = draw[draw["vessel"] == v]
        if d.empty:
            continue
        cost = np.linalg.norm(b[["cx", "cy", "cz"]].to_numpy()[:, None]
                              - d[["cx", "cy", "cz"]].to_numpy()[None], axis=2)
        r, c = linear_sum_assignment(cost)
        for i, j in zip(r, c):
            if cost[i, j] <= MATCH_MM:
                out[int(b.index[i])] = int(d.index[j])
    return out


LESION_COLS = ("cx", "cy", "cz", "n_rois", "total_area_mm2", "volume_mm3", "mean_hu_weighted", "max_hu")
PARITY_COLS = ("n_rois", "total_area_mm2", "volume_mm3", "mean_hu_weighted", "max_hu")


def one_patient(pid: str, excluded: set, calib_path: str, n_draws: int,
                config_path: str | None) -> list[dict]:
    """Baseline and per-draw lesions of one patient. Rows: kind 'base'
    (draw -1), per draw one row per baseline lesion ('matched' or 'lost'),
    and one row per draw lesion that matched no baseline lesion ('added')."""
    from predict.features.canonical import compute_canonical_features, grouping_from_config
    from predict.io import load_patient_metadata, parse_calcium_xml
    from predict.stability.realistic import Calibration, jitter_parse_result, sample_draw
    cfg = load_config(Path(config_path) if config_path else None)
    calib = Calibration.from_json(Path(calib_path))
    meta = load_patient_metadata(pid, cfg.paths.data_raw.parent)
    parse = parse_calcium_xml(pid, cfg.paths.data_xml)
    grouping = grouping_from_config(cfg)
    _, _, lpv = compute_canonical_features(parse, meta, excluded, **grouping)
    base = pd.DataFrame(_flatten(pid, lpv))
    rows = base.assign(draw=-1, base_row=range(len(base)), kind="base", matched=True).to_dict("records")
    for idx in range(n_draws):
        draw = sample_draw(pid, "primary", idx, calib, float(meta.pixel_spacing[0]))
        pj = jitter_parse_result(parse, draw.m1_rms_px, draw.m1_seed, excluded)
        _, _, lpv_d = compute_canonical_features(pj, meta, excluded, **grouping)
        dl = pd.DataFrame(_flatten(pid, lpv_d))
        m = _match(base, dl) if len(dl) else {}
        common = {"pid": pid, "draw": idx, "n_lesions_draw": len(dl),
                  "n_lesions_base": len(base), "m1_rms_px": draw.m1_rms_px}
        for bi in range(len(base)):
            rec = {**common, "vessel": base.at[bi, "vessel"], "lesion_idx": int(base.at[bi, "lesion_idx"]),
                   "base_row": bi, "matched": bi in m, "kind": "matched" if bi in m else "lost"}
            if bi in m:
                rec.update({c: dl.at[m[bi], c] for c in LESION_COLS})
            rows.append(rec)
        for dj in sorted(set(range(len(dl))) - set(m.values())):
            rows.append({**common, "vessel": dl.at[dj, "vessel"], "lesion_idx": -1 - dj,
                         "base_row": -1, "matched": False, "kind": "added",
                         **{c: dl.at[dj, c] for c in LESION_COLS}})
    return rows


def require_baseline_parity(base: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """The unperturbed lesions must equal lesions.csv: same keys, same
    clustered inputs. Raises SystemExit otherwise (D043)."""
    pm = base.merge(ref, on=KEYS, how="outer", suffixes=("", "_ref"), indicator=True)
    both = pm["_merge"] == "both"
    if not both.all():
        raise SystemExit(f"hard stop: baseline lesion keys differ from lesions.csv "
                         f"({int((pm['_merge'] == 'left_only').sum())} only in the replay, "
                         f"{int((pm['_merge'] == 'right_only').sum())} only in lesions.csv)")
    bad = [c for c in PARITY_COLS
           if not np.allclose(pm[c].astype(float), pm[f"{c}_ref"].astype(float),
                              rtol=1e-9, atol=1e-9, equal_nan=True)]
    if bad:
        raise SystemExit(f"hard stop: baseline lesions differ from lesions.csv in {bad}")
    return {"n_lesions": int(len(pm)), "columns_checked": list(PARITY_COLS)}


def c8_jaccard(base_c8: set, draw: pd.DataFrame, c8: int) -> float:
    """Jaccard of the baseline C8 set and one draw's C8 set. Matched draw
    lesions keep their baseline key; added lesions get their own keys, so a
    split C8 lesion counts as an addition, and a lost C8 lesion as a loss."""
    keys = draw[KEYS].astype(str).apply(tuple, axis=1)
    draw_c8 = set(keys[draw["cluster_draw"].to_numpy() == c8])
    u = base_c8 | draw_c8
    return len(base_c8 & draw_c8) / len(u) if u else float("nan")


def main() -> int:
    from joblib import Parallel, delayed
    from predict.figures._lesion_classes import discover_c8_like_cluster
    from predict.stability.icc import icc_1_1
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--limit", type=int, default=None, help="first N patients (smoke)")
    ap.add_argument("--n-jobs", type=int, default=default_n_jobs())
    ap.add_argument("--random-state", type=int, default=42,
                    help="seed of the lesion k-means (lesion_morphology/run.py --random-state)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # The calibration of THIS outputs root, and only the one stage 4b
    # recorded (D043).
    calib_path = cfg.paths.outputs / "04b_calibration" / "calibration.json"
    rec4b = cfg.paths.outputs / "04b_realistic" / "COMPLETE.json"
    if not rec4b.exists() or json.loads(rec4b.read_text(encoding="utf-8")).get(
            "calibration_sha") != file_sha(calib_path):
        raise SystemExit(f"hard stop: {calib_path} is not the calibration recorded in {rec4b}")
    lm_dir = stage9_root(cfg) / "lesion_morphology"
    out_dir = lm_dir / ("lesion_stability_d033" + ("_smoke" if args.limit else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    pids = pd.read_csv(cfg.paths.outputs / "06_reduce" / "cohort_metadata.csv", dtype={"pid": str})["pid"].tolist()
    if args.limit:
        pids = pids[: args.limit]
    trips = pd.read_csv(cfg.paths.outputs / "02_preprocessed" / "xml_roundtrip.csv", dtype={"pid": str})
    fails = trips[(~trips["passes"].astype(bool)) & (trips["matched_via"] != "dirty")]
    excl: dict[str, set] = {}
    for r in fails.itertuples():
        excl.setdefault(str(r.pid), set()).add((int(r.image_index), int(r.roi_idx_in_slice)))

    res = Parallel(n_jobs=args.n_jobs)(
        delayed(one_patient)(p, excl.get(p, set()), str(calib_path), args.n_draws,
                             str(args.config) if args.config else None) for p in pids)
    allrows = pd.DataFrame([r for rows in res for r in rows])
    allrows.to_csv(out_dir / "lesion_draws.csv", index=False)

    base = allrows[allrows.kind == "base"].copy()
    draws = allrows[allrows.kind.isin(["matched", "lost"])].copy()   # one row per baseline lesion
    added = allrows[allrows.kind == "added"].copy()

    # 1. parity against lesions.csv: enforced before anything else
    ref_all = pd.read_csv(cfg.paths.outputs / "03_features" / "lesions.csv", dtype={"pid": str})
    ref = ref_all[ref_all.pid.isin(pids)]
    parity = require_baseline_parity(base, ref)

    # 2. grouping stability
    per_draw_match = draws.groupby("draw")["matched"].mean()
    n_added = added.groupby("draw").size().reindex(per_draw_match.index, fill_value=0)
    always = draws.groupby(KEYS)["matched"].all()
    always_keys = always[always].index

    # 3. ICC on lesions matched in every draw
    dm = draws.set_index(KEYS).loc[always_keys].reset_index()
    tf = transform(dm)
    icc_rows = []
    for f in FEATURES:
        mat = (pd.concat([dm[KEYS + ["draw"]], tf[[f]]], axis=1)
               .pivot_table(index=KEYS, columns="draw", values=f).to_numpy())
        icc, lo, hi = icc_1_1(mat)
        icc_rows.append({"feature": f, "icc": icc, "ci_low": lo, "ci_high": hi,
                         "n_lesions": int(mat.shape[0]), "passes": bool(icc >= ICC_MIN)})
    icc_df = pd.DataFrame(icc_rows)
    icc_df.to_csv(out_dir / "icc.csv", index=False)
    hu = dm.merge(base[KEYS + ["max_hu", "mean_hu_weighted"]], on=KEYS, suffixes=("", "_b"))
    # max_hu is the XML ROI maximum (no M1 path); mean_hu_weighted is the
    # area-weighted mean of XML ROI means, so M1 moves it through the weights.
    hu_diff = {c: float(np.nanmax(np.abs(hu[c] - hu[f"{c}_b"]))) for c in ("max_hu", "mean_hu_weighted")}

    # 4. cluster stability vs the stored k-means partition, rebuilt exactly
    stored = pd.read_csv(lm_dir / "lesion_cluster_labels.csv", dtype={"pid": str})
    col = [c for c in stored.columns if c.startswith("cluster_kmeans_k")][0]
    k = int(col.rsplit("k", 1)[1])
    morph = pd.read_csv(lm_dir / "lesion_features.csv", dtype={"pid": str})
    zcols = [c for c in morph.columns if c not in KEYS]
    if zcols != list(FEATURES):
        raise SystemExit(f"hard stop: lesion_features.csv columns {zcols} are not {list(FEATURES)}")
    # run.py z-scored over every lesion in lesions.csv (after the NaN drop), not only the cohort
    raw_ref = ref_all.dropna(subset=["volume_mm3", "total_area_mm2", "mean_hu_weighted", "max_hu", "n_rois"])
    tref = transform(raw_ref)
    mu, sd = tref.mean(), tref.std(ddof=1)
    z_ref = ((tref[list(FEATURES)] - mu) / sd).assign(**{c: raw_ref[c] for c in KEYS})
    z_chk = morph.merge(z_ref, on=KEYS, how="outer", suffixes=("", "_re"), indicator=True)
    if not (z_chk["_merge"] == "both").all() or not np.allclose(
            z_chk[list(FEATURES)].to_numpy(float),
            z_chk[[f"{c}_re" for c in FEATURES]].to_numpy(float), rtol=1e-9, atol=1e-9):
        raise SystemExit("hard stop: the replayed z-scoring does not reproduce lesion_features.csv")
    lab = morph[KEYS].merge(stored[KEYS + [col]], on=KEYS, how="left", validate="one_to_one")[col]
    if lab.isna().any():
        raise SystemExit("hard stop: lesion_features.csv lesions without a stored cluster label")
    lab = lab.to_numpy(int)
    model = fit_kmeans(morph[zcols].to_numpy(float), k, random_state=args.random_state)
    if not np.array_equal(model.labels_, lab):
        raise SystemExit(f"hard stop: refitting k-means (k={k}, seed {args.random_state}) does not "
                         f"reproduce the stored labels ({int((model.labels_ != lab).sum())} differ)")
    c8 = discover_c8_like_cluster(pd.read_csv(lm_dir / "cluster_profiles.csv"),
                                  pd.read_csv(lm_dir / "cluster_vessel_chi_square.csv"))
    base_lab = stored.set_index(KEYS)[col]

    def assign(df: pd.DataFrame) -> np.ndarray:
        z = ((transform(df)[list(FEATURES)] - mu[list(FEATURES)]) / sd[list(FEATURES)]).to_numpy(float)
        return model.predict(z)

    dmatch = draws[draws.matched].copy()
    dmatch["cluster_draw"] = assign(dmatch)
    dmatch["cluster_base"] = base_lab.reindex(pd.MultiIndex.from_frame(dmatch[KEYS])).to_numpy()
    if dmatch["cluster_base"].isna().any():
        raise SystemExit("hard stop: matched draw lesions without a stored baseline label")
    keep = dmatch.groupby("draw").apply(lambda g: float((g.cluster_draw == g.cluster_base).mean()))
    if len(added):
        added["cluster_draw"] = assign(added)
    # Only lesions of the patients processed in this run (all patients in the full run).
    in_run = stored[KEYS].astype(str).apply(tuple, axis=1).isin(
        set(map(tuple, base[KEYS].astype(str).values)))
    base_c8 = (set(map(tuple, stored.loc[in_run & (stored[col] == c8), KEYS].astype(str).values))
               if c8 is not None else set())
    jac = pd.Series(
        [c8_jaccard(base_c8, pd.concat([dmatch[dmatch.draw == d], added[added.draw == d]]), c8)
         if c8 is not None else np.nan for d in per_draw_match.index],
        index=per_draw_match.index)

    stable_morph = bool(icc_df.passes.all())
    c8_med = float(np.nanmedian(jac)) if jac.notna().any() else float("nan")
    c8_verdict = ("NOT EVALUABLE (no C8 lesions in these patients)" if not base_c8
                  else "STABLE" if c8_med >= C8_JACCARD_MIN else "NOT STABLE")
    L = ["=" * 92, f"LESION STABILITY UNDER D033 M1 JITTER ({len(pids)} patients, {args.n_draws} draws)", "=" * 92,
         f"parity vs lesions.csv: {parity['n_lesions']} lesions, all keys and "
         f"{', '.join(parity['columns_checked'])} equal (enforced); stored labels rebuilt exactly",
         f"grouping: per-draw matched fraction median {per_draw_match.median():.3f} "
         f"[{per_draw_match.min():.3f}-{per_draw_match.max():.3f}]; lesions matched in every draw "
         f"{len(always_keys)}/{len(base)}; lesions created by regrouping per draw median "
         f"{int(n_added.median())} [{int(n_added.min())}-{int(n_added.max())}]",
         "", "ICC(1,1) over draws (lesions matched in every draw):"]
    for r in icc_df.itertuples():
        L.append(f"  {r.feature:<26} {r.icc:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]  n={r.n_lesions}  "
                 f"{'PASS' if r.passes else 'FAIL'}")
    L.append(f"HU max |draw - base|: max_hu {hu_diff['max_hu']:.3g}, mean_hu_weighted "
             f"{hu_diff['mean_hu_weighted']:.3g} HU (XML ROI values; they move when M1 changes ROI areas "
             f"(mean) or regroups ROIs into different lesions (both))")
    L.append(f"cluster kept per draw: median {keep.median():.3f} [{keep.min():.3f}-{keep.max():.3f}]")
    if base_c8:
        L.append(f"C8-like = C{c8}: per-draw Jaccard median {c8_med:.3f} "
                 f"[{np.nanmin(jac):.3f}-{np.nanmax(jac):.3f}] (base C8 lesions {len(base_c8)})")
    else:
        L.append(f"C8-like = C{c8}: no C8 lesions among the processed patients")
    L.append(f"\nRULES (fixed before running): morphology every ICC >= {ICC_MIN} -> "
             f"{'STABLE' if stable_morph else 'NOT STABLE'}; C8 median Jaccard >= {C8_JACCARD_MIN} -> "
             f"{c8_verdict}")
    L.append("=" * 92)
    text = "\n".join(L)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    pd.DataFrame({"draw": jac.index, "c8_jaccard": jac.values,
                  "cluster_kept": keep.reindex(jac.index).values,
                  "matched_fraction": per_draw_match.reindex(jac.index).values,
                  "n_added": n_added.reindex(jac.index).values}).to_csv(
        out_dir / "per_draw.csv", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
