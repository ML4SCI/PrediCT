import os
import pandas as pd
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    RandRotate90d, RandCropByPosNegLabeld,
    RandFlipd, SpatialPadd, ToTensord, RandGaussianNoised,
    CenterSpatialCropd, ScaleIntensityRanged, CropForegroundd
)
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import CastToTyped, AsDiscreted
import nibabel as nib


# Shared deterministic preprocessing for training and validation
def get_base_transforms():
    return [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        # Crop the foreground before intensity scaling
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            select_fn=lambda x: x > -200,
            margin=10
        ),

        # Normalize CT intensities from [-100, 1000] to [0, 1]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
    ]


def get_transforms(patch_size):
    print("🚀 LOADER CHECK: THE CODE IS ACTUALLY RUNNING!")

    train_transforms = Compose([
        *get_base_transforms(),

        # Sample positive and negative patches
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 32),
            pos=1,
            neg=1,
            num_samples=4
        ),

        ToTensord(keys=["image", "label"]),

        CastToTyped(
            keys=["image", "label"],
            dtype=(torch.float32, torch.uint8)
        ),

        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode="constant"
        ),

        AsDiscreted(keys=["label"], threshold=0.5),

        # Training-only augmentations
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.5
        ),

        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3
        ),

        ToTensord(keys=["image", "label"]),

        RandGaussianNoised(
            keys=["image"],
            prob=0.2,
            mean=0.0,
            std=0.1
        ),
    ])

    # Validation uses the same deterministic preprocessing
    # without random sampling or augmentation
    val_transforms = Compose([
        *get_base_transforms(),

        SpatialPadd(
            keys=["image", "label"],
            spatial_size=patch_size,
            mode="constant"
        ),

        AsDiscreted(keys=["label"], threshold=0.5),
        ToTensord(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms


def has_calcium(file_dict):
    lbl_img = nib.load(file_dict['label'])
    return lbl_img.get_fdata().max() > 0


def get_coca_loaders(
    train_parquet,
    val_parquet,
    patch_size=(128, 128, 32),
    batch_size=2
):
    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)

    train_files = [
        {"image": r["image_path"], "label": r["mask_path"]}
        for _, r in train_df.iterrows()
    ]

    val_files = [
        {"image": r["image_path"], "label": r["mask_path"]}
        for _, r in val_df.iterrows()
    ]

    train_trans, val_trans = get_transforms(patch_size)

    # Separate positive and negative samples
    pos_files = [f for f in train_files if has_calcium(f)]
    neg_files = [f for f in train_files if f not in pos_files]

    # Curated training dataset
    curated_train_files = pos_files + neg_files[:int(len(pos_files) * 0.1)]

    train_ds = CacheDataset(
        data=curated_train_files,
        transform=train_trans,
        cache_rate=1.0
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_trans,
        cache_rate=1.0
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=4
    )

    return train_loader, val_loader