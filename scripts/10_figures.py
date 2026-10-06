#!/usr/bin/env python
"""Figure builder for the PrediCT paper + supplementary.

Reads all stages 5-7 outputs + the lesion-morphology experiment outputs
and produces 11 figures (6 main + 5 supplementary) at 300 dpi PDF + PNG.

Outputs are written to ``outputs/figures.partial/`` and published to
``outputs/figures/`` with ``COMPLETE_stage10.json`` (inputs: the stage 6, 7
and 9 records; code fingerprint; output hashes) only when every figure built
(D044). A failing figure exits 1 and publishes nothing. ``--only`` / ``--skip`` runs write to ``outputs/figures_smoke/``
(PARTIAL_SMOKE_DO_NOT_USE) and never replace the published set.

Usage:
    python scripts/10_figures.py
    python scripts/10_figures.py --only F1 F3      # smoke folder only
    python scripts/10_figures.py --skip SF5        # smoke folder only
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from datetime import datetime, timezone

from predict.completion import code_sha, digest, file_sha, files_manifest, write_marker
from predict.config import REPO_ROOT, load_config
from predict.provenance import code_revision
from predict.publish import fresh_partial, publish
from predict.figures import style
from predict.figures.figure_1_patient_pca import build as build_f1
from predict.figures.figure_2_gap_grid import build as build_f2
from predict.figures.figure_3_spatial_burden_split import build as build_f3
from predict.figures.figure_4_burden_composition import build as build_f4
from predict.figures.figure_5_c8_anatomy import build as build_f5
from predict.figures.figure_6_phenotype_table import build as build_f6
from predict.figures.figure_s1_robust_discriminator_heatmap import build as build_s1
from predict.figures.figure_s2_monotonicity_scatter import build as build_s2
from predict.figures.figure_s3_directional_verdicts import build as build_s3
from predict.figures.figure_s4_hennig_stability import build as build_s4
from predict.figures.figure_s5_lesion_radar import build as build_s5


STAGE10_MARKER = "COMPLETE_stage10.json"


def stage10_fingerprint() -> str:
    return digest({"code": code_sha(extra=[REPO_ROOT / "scripts" / "10_figures.py"],
                                    packages=("figures",))})


def stage10_inputs(outputs: Path) -> dict:
    return {
        "stage6_record_sha": file_sha(outputs / "06_reduce" / "COMPLETE_stage6.json"),
        "stage7_record_sha": file_sha(outputs / "07_analyse" / "COMPLETE_stage7.json"),
        "stage9_record_sha": file_sha(outputs / "exploratory" / "COMPLETE_stage9.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None,
                        help="only build the listed figures (e.g. F1 SF3)")
    parser.add_argument("--skip", nargs="+", default=None,
                        help="skip the listed figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("figures")

    cfg = load_config(args.config)
    outputs = cfg.paths.outputs
    production = args.only is None and args.skip is None
    target = outputs / ("figures" if production else "figures_smoke")

    reduce_dir = outputs / "06_reduce"
    from predict.discover.stage6 import require_stage6_complete
    require_stage6_complete(reduce_dir, cfg)   # figures read stage 6 outputs
    analyse_dir = outputs / "07_analyse"
    lesion_morph_dir = outputs / "exploratory" / "lesion_morphology"
    features_csv = outputs / "03_features" / "features.csv"

    # (figure_id, stem, builder, kwargs)
    figures = [
        ("F1",  "figure_1_patient_pca",              build_f1, {"reduce_dir": reduce_dir}),
        ("F2",  "figure_2_gap_grid",                 build_f2, {"reduce_dir": reduce_dir}),
        ("F3",  "figure_3_spatial_burden_split",     build_f3,
            {"reduce_dir": reduce_dir, "analyse_dir": analyse_dir, "features_csv": features_csv}),
        ("F4",  "figure_4_burden_composition",       build_f4, {"lesion_morph_dir": lesion_morph_dir}),
        ("F5",  "figure_5_c8_anatomy",               build_f5,
            {"lesion_morph_dir": lesion_morph_dir}),
        ("F6",  "figure_6_phenotype_table",          build_f6, {"analyse_dir": analyse_dir}),
        ("SF1", "figure_s1_robust_discriminator_heatmap", build_s1, {"analyse_dir": analyse_dir}),
        ("SF2", "figure_s2_monotonicity_scatter",    build_s2, {"analyse_dir": analyse_dir}),
        ("SF3", "figure_s3_directional_verdicts",    build_s3, {"analyse_dir": analyse_dir}),
        ("SF4", "figure_s4_hennig_stability",        build_s4, {"reduce_dir": reduce_dir}),
        ("SF5", "figure_s5_lesion_radar",            build_s5, {"lesion_morph_dir": lesion_morph_dir}),
    ]

    selected = args.only or [fid for fid, *_ in figures]
    skipped = set(args.skip or [])
    if any("analyse_dir" in kw for fid, _, _, kw in figures
           if fid in selected and fid not in skipped):
        # Figures built from stage 7 outputs need a complete, current record.
        from predict.analyse.stage7 import require_stage7_complete
        require_stage7_complete(analyse_dir, cfg)
    if any("lesion_morph_dir" in kw for fid, _, _, kw in figures
           if fid in selected and fid not in skipped):
        # Figures built from the stage 9 lesion chain need its record (D043).
        from predict.stage9 import require_stage9_complete
        require_stage9_complete(cfg)
    inputs = stage10_inputs(outputs)
    out_dir = fresh_partial(target)
    if not production:
        (out_dir / "PARTIAL_SMOKE_DO_NOT_USE").write_text("partial figure set\n", encoding="utf-8")

    n_ok = 0
    n_fail = 0
    for fid, stem, builder, kwargs in figures:
        if fid not in selected or fid in skipped:
            continue
        try:
            log.info("building %s (%s)...", fid, stem)
            fig = builder(**kwargs)
            paths = style.save(fig, out_dir, stem)
            log.info("  saved %s  +  %s",
                     paths["pdf"].name, paths["png"].name)
            n_ok += 1
        except Exception as e:
            log.error("  FAILED %s: %s", fid, e)
            log.debug(traceback.format_exc())
            n_fail += 1

    if n_fail:
        log.error("done: %d OK, %d FAILED; nothing published (partial set left in %s)",
                  n_ok, n_fail, out_dir)
        return 1
    if production:
        if stage10_inputs(outputs) != inputs:
            raise SystemExit("hard stop: stage 6/7/9 records changed during the run; nothing published")
        write_marker(out_dir, {
            "stage": 10,
            "stage10_fingerprint": stage10_fingerprint(),
            "inputs": inputs,
            "outputs": files_manifest(out_dir, [f.name for f in out_dir.iterdir() if f.is_file()]),
            "revision": code_revision(outputs.parent),
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, name=STAGE10_MARKER)
    publish(out_dir, target)
    log.info("done: %d OK. outputs in %s", n_ok, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
