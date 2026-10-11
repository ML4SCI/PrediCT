"""
config.py — the only settings ReRo-Net inference needs, for RUN_NO_COORD.

model.py reads `model_config` at import (`import config as cfg`). Values are
copied from ../config_snapshot.json (the run's own training snapshot); change
them only together with the checkpoint.
"""

model_config = {
    "MODEL_NAME": "RIU_NET_V1.0",
    "USE_DEEP_SUPERVISION": False,
    "BASE_CHANNELS": 32,
    "NUM_CLASSES": 1,
    "VERBOSE": False,
    "IN_CHANNELS": 2,              # dual HU window only, no coord channels
    "DA_NUM_HEADS": 8,
    "DA_NUM_POINTS": 4,
    "FNO_MODES": (8, 8, 8),
    "ROI_SIZE": (128, 128, 32),
}

train_config = {
    "VAL_THRESHOLD": 0.97,         # sigmoid cut used for the reported test metrics
}

# Inference settings from Segmentation_Rajat/wrapper.py (its defaults, since
# the snapshot has no VAL_SW_* keys) and eval.py's large-lesion filter.
infer_config = {
    "SW_BATCH_SIZE": 4,
    "SW_OVERLAP": 0.25,
    "MAX_VOXEL_THRESHOLD": 15000,  # 3D components larger than this are removed
    "TISSUE_WINDOW": (-100.0, 400.0),
    "CALCIUM_WINDOW": (130.0, 1000.0),
}
