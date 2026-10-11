"""
pipeline_rero.py — ReRo-Net specific inference pipeline

Handles models marked with arch="rero". The model's own code lives in
models/<id>/code/ and runs as a subprocess there, so its `import config`
never mixes with the app's modules — the same isolation pipeline_nnunet.py
gets from the nnUNet CLI.
"""
import sys
import tempfile
import subprocess
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from .pipeline import save_nifti

def predict_rero(image: sitk.Image, manifest: dict) -> np.ndarray:
    """Run ReRo-Net inference on the given prepared image.

    Args:
        image: sitk.Image in RAS orientation, already resampled (and cropped).
        manifest: The loaded model manifest containing weights_path.

    Returns:
        prob: (Z, Y, X) numpy array holding the binary mask as float32 (0/1).
    """
    weights = manifest["weights_path"]
    if not weights.exists():
        raise FileNotFoundError(f"ReRo-Net checkpoint not found at {weights}")
    code_dir = weights.parent / "code"

    with tempfile.TemporaryDirectory() as tmp_dir:
        in_file = Path(tmp_dir) / "ct.nii.gz"
        out_file = Path(tmp_dir) / "pred.nii.gz"
        save_nifti(image, in_file)

        cmd = [sys.executable, "infer.py", str(in_file), str(out_file), str(weights)]
        result = subprocess.run(cmd, cwd=code_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"ReRo-Net prediction failed:\n{result.stdout}")

        if not out_file.exists():
            raise FileNotFoundError(f"ReRo-Net output not found at {out_file}")

        prob = sitk.GetArrayFromImage(sitk.ReadImage(str(out_file)))  # (Z, Y, X)

    expected = sitk.GetArrayViewFromImage(image).shape
    if prob.shape != expected:
        raise ValueError(f"ReRo-Net mask {prob.shape} does not match the CT grid {expected}")

    return prob.astype(np.float32)
