import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import pytest

from monai.transforms import Randomizable

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from data.big_data_loader import get_transforms


@pytest.fixture
def sample(tmp_path):

    image = np.full((160, 160, 80), -1000, dtype=np.float32)
    label = np.zeros(image.shape, dtype=np.uint8)

    image[30:130, 30:130, 20:60] = 200
    image[70:80, 70:80, 30:40] = 2000
    label[70:80, 70:80, 30:40] = 1

    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"

    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)

    return {
        "image": str(image_path),
        "label": str(label_path)
    }


def test_validation_intensity_range(sample):
    _, val = get_transforms((128, 128, 64))
    image = val(sample)["image"]

    assert image.min().item() == 0
    assert image.max().item() == 1


def test_exact_intensity_scaling(sample):
    _, val = get_transforms((128, 128, 64))
    image = val(sample)["image"]

    expected = (200 + 100) / 1100

    assert torch.isclose(image, torch.tensor(expected)).any()


def test_image_label_alignment(sample):
    _, val = get_transforms((128, 128, 64))
    result = val(sample)

    image = result["image"]
    label = result["label"]

    assert image.shape == label.shape
    assert label.sum().item() == 1000
    assert torch.equal(image == 1, label == 1)


def test_foreground_crop_and_padding(sample):
    _, val = get_transforms((128, 128, 64))
    result = val(sample)

    assert result["image"].shape == (1, 128, 128, 64)
    assert result["label"].shape == (1, 128, 128, 64)


def test_deterministic_validation(sample):
    _, val = get_transforms((128, 128, 64))

    first = val(sample)
    second = val(sample)

    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["label"], second["label"])


def test_no_random_validation_transforms():
    _, val = get_transforms((128, 128, 64))

    assert not any(
        isinstance(t, Randomizable)
        for t in val.transforms
    )


def test_shared_preprocessing(sample):
    train, val = get_transforms((128, 128, 64))

    train_base = sample.copy()
    val_base = sample.copy()

    for transform in train.transforms[:4]:
        train_base = transform(train_base)

    for transform in val.transforms[:4]:
        val_base = transform(val_base)

    assert torch.equal(train_base["image"], val_base["image"])
    assert torch.equal(train_base["label"], val_base["label"])