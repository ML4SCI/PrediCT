# Source of this code

Vendored from `ML4SCI/PrediCT`, branch `rajat_segmentation`, commit `43cd95a` (2026-09-12),
folder `Segmentation_Rajat/`.

| File | Origin | Edits |
|---|---|---|
| `model.py` | `Segmentation_Rajat/model.py` | Imports `riu_modules.*` → `rero_modules.*` (the branch's folder is `rero_modules`, so it did not import as committed) |
| `rero_modules/*` | `Segmentation_Rajat/rero_modules/` | `fno3d.py`: `import psutil` moved from the top into its `__main__` benchmark block (unused by the model, not installed here) |
| `config.py` | new | Only `model_config` values from `../config_snapshot.json`, plus the threshold and sliding-window settings `wrapper.py` uses |
| `infer.py` | new | Mirrors `wrapper.py` without the LW-UNet heart ROI. `lesion_post_process` is copied from `eval.py` |
| `../config_snapshot.json` | `Segmentation_Rajat/runs/run_no_coord/config_snapshot.json` | none |

Weights: `../best_dice.pth` from Drive `ReRoNet_Anonymous_Models/RUN_NO_COORD/` (not in git).

Known difference from Rajat's evaluation: his runs zero the CT outside an LW-UNet heart
bounding box and erode-filter predictions to it. Here the app's TotalSegmentator crop is used
instead (manifest `crop: true`), so outputs are close to his numbers but not bit-identical.
