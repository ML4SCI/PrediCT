"""
run.py — Inference orchestrator for PrediCT.

Ties together paths, pipeline, scoring, and rendering into a complete execution flow.
Handles caching of prep stages and writes the provenance `run.json` to the output folder.

Does NOT: handle HTTP requests, instantiate PyTorch models, or implement math.
Called by: CLI entrypoint, server.py.

Every run belongs to one account: inputs and outputs live under
data/users/<user_id>/ (see paths.py). Which account is always explicit.

Usage:
    # Python API
    run(user_id=1, study_id="some_hash", model_id="a1-roi", custom_input=Path("/some/path"))

    # CLI  (account ids: python -m src.backend.accounts list)
    python -m src.backend.run --user 1 --study patient_1 --model a1-roi       # an uploaded scan
    python -m src.backend.run --user 1 --input /path/to/dicom --model a3-coverage-v2
"""
import json
import argparse
from datetime import datetime
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import csv

import sys
# Make sure we can import 'src' even if the script is executed directly from another directory
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.backend.paths import scan_input_dir, work_dir, out_dir, study_id_from_series
from src.backend.registry import load_manifest
from src.backend.pipeline import load, resample, crop_heart, normalize, predict, save_nifti
from src.backend.scoring import score, totals
from src.backend.grouping import lesion_3d_table
from src.backend.render import save_slices
from src.backend.mesh import build_meshes

# Scoring conventions, not model properties — they belong here, not in a
# manifest. Both are written into run.json so any output folder records what
# actually ran.
MIN_AREA_MM2   = 1.0   # clinical minimum lesion, conventionally >=3 contiguous px
MAX_GAP_SLICES = 0     # 3D lesion linking only (grouping.py). 0 = strictly
                       # adjacent slices. Unswept — raise it only on a measured
                       # gap-frequency count, see the 3D-lesion brief.

def write_csv(rows: list[dict], path: Path):
    if not rows:
        with open(path, "w") as f:
            f.write("No lesions found.\n")
        return
    with open(path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def run(user_id: int, study_id: str, model_id: str, crop: bool = None, progress=None, custom_input: Path = None):
    """Execute the inference pipeline for a given study and model of one account."""
    m = load_manifest(model_id)
    crop = m["crop"] if crop is None else crop
    if crop != m["crop"]:
        print(f"NOTE: crop={crop} overrides the {model_id} default (crop={m['crop']}); "
              f"it was trained on {'heart-cropped' if m['crop'] else 'full'} volumes.")

    def p(stage, pct):
        if progress:
            progress(stage, pct)

    # Cropped and full volumes are different inputs, so each has its own cache.
    # One shared cache let a model silently reuse a CT prepared the other way.
    w = work_dir(user_id, study_id) / ("crop" if crop else "full")
    w.mkdir(parents=True, exist_ok=True)
    ct_path = w / "ct.nii.gz"
    # The heart mask is cached beside the CT because it is produced by the same
    # (slow) crop stage. If the CT is cached and this is not, the study was
    # prepared before meshing existed: report None rather than re-running
    # TotalSegmentator behind the caller's back or faking a shape.
    heart_path = w / "heart.nii.gz"
    
    image = None
    heart_img = None
    if ct_path.exists():
        p("cached", 0.3)
        image = sitk.ReadImage(str(ct_path))
        # Note: cached image already has RAS orientation applied from prior prep.
        if heart_path.exists():
            heart_img = sitk.ReadImage(str(heart_path))
        else:
            print("NOTE: no cached heart mask for this study — 3D view will "
                  f"show lesions without the heart shell. Delete {w} "
                  "and re-run to generate one.")
    else:
        p("load", 0.05)
        load_path = custom_input if custom_input else scan_input_dir(user_id, study_id)
        image = load(load_path)
        
        p("resample", 0.15)
        image = resample(image, tuple(m["spacing"]))
        
        if crop:
            p("crop", 0.3)
            image, heart_img = crop_heart(image, m["margin_mm"], fast=m.get("ts_fast", False))
        else:
            print("Cropping OFF — the full field of view is used.")
            
        # Orient to RAS AFTER cropping to match training pipeline order!
        print("Reorienting to RAS...")
        image = sitk.DICOMOrient(image, m.get("orientation", "RAS"))
        # The mask must take the identical reorientation or the heart shell
        # will be mirrored relative to the lesions in the 3D view.
        if heart_img is not None:
            heart_img = sitk.DICOMOrient(heart_img, m.get("orientation", "RAS"))
        
        # Ensure spacing didn't permute in a way that alters the volume area calculations
        spc = image.GetSpacing()
        req_spc = tuple(m["spacing"])
        assert np.allclose(spc, req_spc, atol=1e-4), f"RAS permuted axes: {spc}"
        
        # render.py's flips assume RAS (diag(-1,-1,1)). DICOMOrient guarantees it, but
        # a wrong direction matrix would silently mirror the display, not error.
        assert np.allclose(image.GetDirection(), (-1,0,0, 0,-1,0, 0,0,1), atol=1e-6), \
            f"expected RAS direction cosines, got {image.GetDirection()}"
        
        sitk.WriteImage(image, str(ct_path))
        if heart_img is not None:
            sitk.WriteImage(heart_img, str(heart_path))

    array = sitk.GetArrayFromImage(image)                 # (Z, Y, X)
    
    p("predict", 0.5)
    if m.get("arch") == "nnunet":
        from src.backend.pipeline_nnunet import predict_nnunet
        prob = predict_nnunet(image, m) # returns (Z, Y, X)
    elif m.get("arch") == "rero":
        from src.backend.pipeline_rero import predict_rero
        prob = predict_rero(image, m) # returns (Z, Y, X)
    else:
        x = normalize(np.transpose(array, (2, 1, 0)), tuple(m["hu_window"])) # (X, Y, Z)
        prob_xyz = predict(
            x, 
            m["weights_path"], 
            m["arch"],
            tuple(m["patch"]), 
            m["overlap"], 
            m["activation"]
        )
        prob = np.transpose(prob_xyz, (2, 1, 0)) # back to (Z, Y, X)
    
    o = out_dir(user_id, study_id, model_id)
    o.mkdir(parents=True, exist_ok=True)
    
    prob_img = sitk.GetImageFromArray(prob)
    prob_img.CopyInformation(image)
    
    # Save NIfTIs in the RAS orientation
    save_nifti(prob_img, o / "pred.nii.gz")
    save_nifti(image, o / "ct.nii.gz")

    p("score", 0.8)
    rows = score(prob, array, image.GetSpacing(), m["output"], m["threshold"],
                 min_area_mm2=MIN_AREA_MM2, max_gap_slices=MAX_GAP_SLICES)
    groups = lesion_3d_table(rows, image.GetSpacing()[2])
    summary = totals(rows)

    # Grouping is descriptive. If it ever changes a number, something has leaked
    # from grouping.py into the area or weight terms — stop rather than publish.
    assert abs(sum(g["total_agatston"] for g in groups) - summary["agatston_total"]) < 1e-9, \
        "3D rollup disagrees with the per-slice total — grouping has affected scoring"
    assert sum(g["n_components"] for g in groups) == len(rows), \
        "3D rollup lost or duplicated a component"

    write_csv(rows, o / "lesions.csv")
    write_csv(groups, o / "lesions_3d.csv")
    
    run_provenance = {
        **summary,
        "study_id": study_id,
        "model_id": model_id,
        "cropped": crop,
        "crop_default": m["crop"],   # the model's own setting; differs => overridden
        "crop_margin_mm": m["margin_mm"] if crop else None,
        "shape": list(array.shape),
        "output": m["output"],
        "threshold": m["threshold"],
        "min_area_mm2": MIN_AREA_MM2,
        "max_gap_slices": MAX_GAP_SLICES,
        "hu_window": m["hu_window"],
        "spacing": list(image.GetSpacing()),
        "sha256": m.get("sha256"),
        "locator_version": m.get("locator_version"),
        "date": datetime.now().isoformat()
    }
    
    p("mesh", 0.88)
    heart_array = sitk.GetArrayFromImage(heart_img) if heart_img is not None else None
    if heart_array is not None:
        assert heart_array.shape == array.shape, \
            f"heart mask {heart_array.shape} does not share the CT grid {array.shape}"
    run_provenance["mesh"] = build_meshes(prob, heart_array, image.GetSpacing(),
                                          m["output"], o)

    with open(o / "run.json", "w") as f:
        json.dump(run_provenance, f, indent=2)
        
    p("render", 0.9)
    save_slices(array, prob, o / "slices", m, rows)
    
    p("done", 1.0)
    return o

if __name__ == "__main__":
    import sys
    
    # MANUAL RUN BLOCK (Triggered if no CLI arguments are provided)
    if len(sys.argv) == 1:
        print("No CLI arguments provided. Running in MANUAL mode...")
        
        # EDIT THESE VALUES FOR MANUAL RUNS:
        MANUAL_USER_ID = 1   # the account that owns the result (python -m src.backend.accounts list)
        MANUAL_INPUT_PATH = Path("/pscratch/sd/s/soham95/SOHAM/coca_raw/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/342/Pro_Gated_Calcium_Score_(CS)_3.0_Qr36_2_BestDiast_74_%")
        MANUAL_MODEL_ID = "a3-coverage-v2"

        study_id = MANUAL_INPUT_PATH.parent.name # Usually the patient ID folder

        def log_progress(stage, pct):
            print(f"[{pct*100:3.0f}%] {stage}")

        print(f"Starting manual run for study '{study_id}' with model '{MANUAL_MODEL_ID}'...")
        out_path = run(MANUAL_USER_ID, study_id, MANUAL_MODEL_ID, progress=log_progress, custom_input=MANUAL_INPUT_PATH)
        print(f"Done. Outputs saved to {out_path}")
        sys.exit(0)

    # EXISTING CLI LOGIC
    from src.backend.accounts import user_exists
    from src.backend.paths import safe_name

    parser = argparse.ArgumentParser(description="Run PrediCT CAC inference pipeline.")
    parser.add_argument("--user", type=int, required=True,
                        help="Account id that owns the result (python -m src.backend.accounts list)")
    parser.add_argument("--study", help="Study ID: an uploaded scan of that account, or the result name with --input")
    parser.add_argument("--input", help="Absolute path to a DICOM directory (instead of an uploaded scan)")
    parser.add_argument("--model", required=True, help="Model ID (e.g., a1-roi)")
    parser.add_argument("--crop", action="store_true", default=None, help="Force crop=True")
    parser.add_argument("--no-crop", dest="crop", action="store_false", help="Force crop=False")

    args = parser.parse_args()
    if not user_exists(args.user):
        parser.error(f"no account with id {args.user} (see: python -m src.backend.accounts list)")

    if args.input:
        study_id = args.study if args.study else Path(args.input).name
        input_path = Path(args.input)
    elif args.study:
        study_id = args.study
        input_path = scan_input_dir(args.user, args.study)
    else:
        parser.error("Must provide either --study or --input")
    try:
        study_id = safe_name(study_id)     # becomes data/users/<id>/out/<study_id>
    except ValueError as e:
        parser.error(str(e))

    print(f"Starting run for study '{study_id}' with model '{args.model}'...")
    def log_progress(stage, pct):
        print(f"[{pct*100:3.0f}%] {stage}")

    out_path = run(args.user, study_id, args.model, crop=args.crop, progress=log_progress, custom_input=input_path)
    print(f"Done. Outputs saved to {out_path}")
