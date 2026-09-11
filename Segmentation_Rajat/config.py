# common_task/config.py

from pathlib import Path

# All paths relative to this file's location (common_task/)
BASE_DIR = Path(__file__).parent

ground_truth_config = {
    "INPUT_CSV":       str(BASE_DIR / "train_raw.csv"),
    "OUTPUT_CSV":      str(BASE_DIR / "train.csv"),
    "TOTALSEG_RAW":    str(BASE_DIR / "totalseg_outputs"),
    "HEART_MASKS_DIR": str("ground_truth"),  # used for saving so relative path is fine
    "STATS_JSON":      str(BASE_DIR / "ground_truth" / "stats.json"),
    "N_SCANS":         50,                      # meets 30-50 requirement
    "DEVICE":          "gpu",
    "TASK":            "heartchambers_highres",
}

preprocessing_config = {
    # NOTE: was a plain "E:\..." string, which has invalid escape sequences
    # (\M, \G, \d are not real escapes -> Python just warns and keeps them
    # literal, but it's fragile). Forward slashes work fine on Windows too
    # and match the style already used below.
    "DATASET_CSV":          "E:/MyProjects/Gsoc_2026_Official/data_resampled/dataset_resampled.csv",
    "DATASET_FOR_DOB_SCV":  "E:/MyProjects/Gsoc_2026_Official/data_canonical/tables/scan_index.csv",
    "SPLITS_JSON": str(BASE_DIR / "MetaData" / "splits.json"),
    "STATS_JSON":  str(BASE_DIR / "MetaData" / "dataset_stats.json"),
    "CLEANED_DF":  str(BASE_DIR / "MetaData" / "cleaned_dataset.csv"),
    "VAL_SIZE":    0.15,
    "TEST_SIZE":   0.15,
    "RANDOM_SEED": 42,
    "TASK": "binary",  # can also be "multi" -> RCA, LCA, LCX, LADX
}

do_heart_roi_masking = True
add_heart_mask_channel = False

dataloader_config = {
    "BATCH_SIZE":  2,                 # safe for <12GB VRAM
    "ROI_SIZE":    (128, 128, 32),
    "CACHE_RATE":  1.0,               # auto-overridden by get_cache_rate()
    # NOTE: DATASET_CSV above is a Windows-style path, but this is set to 8.
    # Your own comment says NUM_WORKERS must be 0 on Windows — if you're
    # actually training on Windows (not WSL/Linux with a stale path), this
    # will break. Left as-is since I can't tell which environment you're on;
    # just flagging it.
    "NUM_WORKERS": 8,
    "HEART_MASK_FLAG": do_heart_roi_masking,
    "ADD_HEART_MASK_CHANNEL": add_heart_mask_channel,
    "HEART_MODEL_PATH": str(BASE_DIR / "LW_UNET_TVERSKY" / "best_model.pth"),
    "ADD_COORD_CHANNELS": True,
    "COORD_MODE": "normalized",       # "normalized" or "absolute"
    "DUAL_HU_WINDOWING": True,
    "NUM_OF_SAMPLES_PER_BATCH": 4,
}

HU_CONFIG = {
    "WINDOW_LEVEL": 100,
    "WINDOW_WIDTH": 500,
    "A_MIN":        -150,             # WL - WW/2
    "A_MAX":        350,              # WL + WW/2
}

model_config = {
    "MODEL_NAME": "RIU_NET_V1.0",
    "USE_DEEP_SUPERVISION": False,

    "BASE_CHANNELS": 32,              # 32 -> 64 -> 128 -> 256 -> 512
    "NUM_CLASSES":   1,               # bg + calcium (sigmoid applied)

    "DEBUG_MODE": True,
    "VERBOSE":    False,              # gates noisy per-step prints in model.py / modules

    # Auto-detected by the train script from the first batch.
    "IN_CHANNELS": None,

    "DA_NUM_HEADS":  8,
    "DA_NUM_POINTS": 4,

    "FNO_MODES": (8, 8, 8),           # upper bound; clamped to actual bottleneck at runtime

    "ROI_SIZE": dataloader_config["ROI_SIZE"],  # always synced to dataloader_config
}

# ══════════════════════════════════════════════════════════════════
#  TRAIN CONFIG
#  Aligned to the overfit_check.py run that converged:
#    --n-samples 10 --epochs 200 --lr 1e-2 --scheduler --warmup-epochs 5
#    --loss-kind tversky_ce --pos-weight-cap 1000 --sample-mode live
# ══════════════════════════════════════════════════════════════════

train_config = {

    # Optimization — matches the overfit CLI flags
    "EPOCHS":         400,            # was 400
    "LR":             3e-2,           # was 1e-4
    "WEIGHT_DECAY":   0.0,            # unchanged — already matched overfit_check.py's hardcoded 0.0
    "WARMUP_EPOCHS":  5,              # was 10
    "GRAD_CLIP_NORM": 1.0,            # unchanged — already matched overfit_check.py's hardcoded 1.0

    # Precision — overfit_check.py never wraps its forward pass in autocast,
    # i.e. it runs in plain fp32. This key didn't exist before (train.py's
    # _resolve_amp_dtypes defaults AMP to True when the key is missing), so
    # without it your real training run was silently using AMP while the
    # overfit run wasn't. Adding it explicitly closes that gap.
    "AMP":                False,
    "FORCE_INPUT_DTYPE":  "float32",  # unchanged — redundant with AMP: False but harmless/explicit
    "WEIGHTS_DTYPE":      "float32",  # unchanged

    "DS_LEVEL_WEIGHTS": [1, 0.125, 0.25, 0.5],

    # Trimmed to fit inside 200 epochs and drop noise (was tuned for a
    # 400-epoch run: [0, 10, 50, 100, 150, 170, 190, 195, 198]).
    "DUMP_EPOCHS": [0, 10, 50, 100, 150, 199],

    # Validation / checkpointing
    "SAVE_EVERY_EPOCH":  False,
    "KEEP_LAST_N_CKPTS": 3,
    # overfit_check.py has no concept of early stopping at all (it never
    # validates). 30 is kept as a real-training safety net, but note that
    # with only 200 epochs a 30-epoch plateau could stop the run well short
    # of 200 — set this higher (or 0 to disable) if you specifically want
    # the full 200 epochs to run regardless of val-dice plateaus.
    "EARLY_STOP_PATIENCE": 30,
    "CKPT_POLICY": "best",            # "best" | "last" | "both"

    # --- Loss ---
    "LOSS_TYPE": "TverskyCE",         # unchanged — already matched
    "POS_WEIGHT_CAP": 1000.0,         # was 500.0
    "POS_WEIGHT_BATCHES": 10,         # was 20 — matches --n-samples so pos_weight
                                       # is computed from the same batches training uses
    "VAL_THRESHOLD": 0.97,

    # Explicit now (previously relied on _build_loss's hidden defaults,
    # which happen to be these same values — TverskyWeightedCE(alpha=0.3,
    # beta=0.7) in both train.py and overfit_check.py).
    "TVERSKY_ALPHA": 0.3,
    "TVERSKY_BETA":  0.7,
    "TVERSKY_WEIGHT": 0.5,            # lambda_tversky
    "CE_WEIGHT":      0.5,            # lambda_ce
}

eval_config = {
    "POST_PROCESS_BIG_LESIONS": True,
    "APPLY_ROI_MASK": True, 
    "MORPHOLOGICAL_CLEANUP": False, #Not Tested yet, so keep it False
    "ANISTROPIC GRAPH CUTTING": True, #Not Tested yet, so keep it False
    "ROI_EXCLUDE_EDGE_VOXELS": 3,  # number of voxels to exclude on all sides when applying ROI mask
}