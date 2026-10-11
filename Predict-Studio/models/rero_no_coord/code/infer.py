"""
infer.py — ReRo-Net (RUN_NO_COORD) inference on one prepared CT.

Runs in its own process with cwd = this folder (see src/backend/pipeline_rero.py),
so model.py's `import config` resolves to the config.py next to it and never
to anything in the app.

Input : a NIfTI already resampled, heart-cropped and in RAS by the app.
Output: a uint8 binary CAC mask on exactly the same grid.

Steps follow Segmentation_Rajat/wrapper.py, minus its LW-UNet heart ROI:
dual HU window -> RIUnet sliding window -> sigmoid > 0.97 -> drop huge blobs.

Usage:
    python infer.py <in.nii.gz> <out.nii.gz> <checkpoint.pth>
"""
import sys
import numpy as np
import SimpleITK as sitk
import torch
import cc3d
from monai.inferers import sliding_window_inference

import config as cfg
from model import RIUnet


def dual_hu_window(hu: np.ndarray) -> np.ndarray:
    """Two channels, as dataset.py DualHUWindowingd: tissue and calcium windows to [0, 1]."""
    out = np.empty((2, *hu.shape), dtype=np.float32)
    for c, (lo, hi) in enumerate((cfg.infer_config["TISSUE_WINDOW"], cfg.infer_config["CALCIUM_WINDOW"])):
        out[c] = (np.clip(hu, lo, hi) - lo) / (hi - lo)
    return out


def lesion_post_process(binary_mask: np.ndarray, max_voxel_threshold: int = 15000,
                        connectivity: int = 26) -> np.ndarray:
    """Removes 3D components larger than max_voxel_threshold (copied from eval.py)."""
    binary_input = (binary_mask > 0).astype(np.uint8)
    if np.sum(binary_input) == 0:
        return binary_input

    labels_out = cc3d.connected_components(binary_input, connectivity=connectivity)
    voxel_counts = cc3d.statistics(labels_out)["voxel_counts"]

    large_component_ids = np.where(voxel_counts > max_voxel_threshold)[0]
    large_component_ids = large_component_ids[large_component_ids != 0]

    cleaned_binary = binary_input.copy()
    if len(large_component_ids) > 0:
        print(f"Removing {len(large_component_ids)} components over {max_voxel_threshold} voxels.")
        cleaned_binary[np.isin(labels_out, large_component_ids)] = 0
    return cleaned_binary


def main(in_path: str, out_path: str, ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ReRo-Net inference on {device}")

    image = sitk.ReadImage(in_path)
    hu = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0)).astype(np.float32)  # (X, Y, Z)
    x = torch.from_numpy(dual_hu_window(hu))[None].to(device)                       # (1, 2, X, Y, Z)

    model = RIUnet(in_channels=cfg.model_config["IN_CHANNELS"]).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    model.eval()

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=x,
            roi_size=tuple(cfg.model_config["ROI_SIZE"]),
            sw_batch_size=cfg.infer_config["SW_BATCH_SIZE"],
            overlap=cfg.infer_config["SW_OVERLAP"],
            predictor=model,
            mode="gaussian",
            device=device,
        )
        probs = torch.sigmoid(logits)
        mask = (probs[0, 0] > cfg.train_config["VAL_THRESHOLD"]).cpu().numpy().astype(np.uint8)

    mask = lesion_post_process(mask, max_voxel_threshold=cfg.infer_config["MAX_VOXEL_THRESHOLD"])

    out = sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)))  # back to (Z, Y, X)
    out.CopyInformation(image)
    sitk.WriteImage(out, out_path)
    print(f"Saved {out_path} ({int(mask.sum())} positive voxels)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python infer.py <in.nii.gz> <out.nii.gz> <checkpoint.pth>")
    main(*sys.argv[1:])
