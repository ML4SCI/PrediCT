"""
train.py — Train RIUnet (CAC segmentation) with per-run folder layout.

CLI:
    python Segmentation_Rajat/train.py EXPERIMENT_NAME
    python Segmentation_Rajat/train.py EXPERIMENT_NAME --smoke-test
    python Segmentation_Rajat/train.py EXPERIMENT_NAME --resume

Run-folder layout (created at <BASE_DIR>/runs/EXPERIMENT_NAME/):
    config_snapshot.json
    ckpts/
        best_dice.pth
        last.pth
        epoch_{N:03d}.pth
    logs/train.log
    attn_maps/enc_L{0..3}/step_*.npy     <- Deformable Attention dumps
    attn_maps/dec_L{0..3}/step_*.npy
    fno_maps/step_*_{before,after}_spatial.npy
    eval_preds/{split}/{id}/image.nii.gz
                     {id}/label.nii.gz
                     {id}/prediction.nii.gz
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from monai.inferers import sliding_window_inference

import config as cfg

# Reuse the dataset + model that already live in this folder
from dataset import build_dataloaders
from model import RIUnet

# Replace imports at top:
from torch.amp import GradScaler, autocast

import os   
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ══════════════════════════════════════════════════════════════════
#  Paths / logging
# ══════════════════════════════════════════════════════════════════

BASE_DIR     = Path(__file__).resolve().parent
REPO_ROOT    = BASE_DIR.parent
RUNS_DIR     = BASE_DIR / "runs"
SPLITS_JSON  = Path(cfg.preprocessing_config["SPLITS_JSON"])


def setup_run_dirs(experiment_name: str) -> Dict[str, Path]:
    """Create (or reuse) the per-run folder layout. Returns a dict of paths."""
    run_dir = RUNS_DIR / experiment_name
    paths = {
        "run":      run_dir,
        "ckpts":    run_dir / "ckpts",
        "logs":     run_dir / "logs",
        "attn":     run_dir / "attn_maps",
        "fno":      run_dir / "fno_maps",
        "eval":     run_dir / "eval_preds",
        "snapshot": run_dir / "config_snapshot.json",
    }

    # Filter out file paths from directory creation
    for k, p in paths.items():
        if k == "snapshot":
            continue
        p.mkdir(parents=True, exist_ok=True)

    return paths


def save_config_snapshot(run_dir: Path) -> None:
    """Dump model_config + dataloader_config + train_config to JSON for reproducibility."""
    snap = {
        "model_config":      cfg.model_config,
        "dataloader_config": cfg.dataloader_config,
        "train_config":      cfg.train_config,
        "hu_config":         cfg.HU_CONFIG,
    }
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(snap, f, indent=2, default=str)


def get_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ══════════════════════════════════════════════════════════════════
#  Reproducibility
# ══════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════
#  Channel auto-detection
# ══════════════════════════════════════════════════════════════════

def auto_detect_in_channels(loader) -> int:
    """Pull one batch from `loader` and return the input channel count."""
    batch = next(iter(loader))
    return int(batch["image"].shape[1])


# ══════════════════════════════════════════════════════════════════
#  Save-dir reset (per-epoch)
# ══════════════════════════════════════════════════════════════════

def reset_all_save_counters(model: nn.Module) -> None:
    """Zero the .npy-file counters on every DeformableAttention3d and SpectralConv3d."""
    for m in model.modules():
        if hasattr(m, "reset_save_counter"):
            m.reset_save_counter()


# ══════════════════════════════════════════════════════════════════
#  Loss
# ══════════════════════════════════════════════════════════════════

def _resize_label_to_logits(label: torch.Tensor, target_hw: tuple, debug: bool = False) -> torch.Tensor:
    """Nearest-neighbour downsample of a label volume to a target spatial size.

    Expects `label` to be (B, 1, D, H, W) (channel-first) — matches the train
    loader output after ToTensord. Returns the same shape at target_hw.
    """
    if label.shape[-3:] == target_hw:
        return label

    if debug:
        print(
            f"  [DEBUG_DS] Resizing label from {label.shape[-3:]} "
            f"to {target_hw} using nearest-neighbour interpolation."
        )

    lbl = F.interpolate(label.float(), size=target_hw, mode="nearest").long()
    return lbl


def ds_loss(
    logits_or_list,
    label: torch.Tensor,
    loss_fn: nn.Module,
    level_weights: List[float],
    debug: bool = False,
) -> torch.Tensor:
    """Average per-level loss weighted by `level_weights`."""

    heads = logits_or_list if isinstance(logits_or_list, (list, tuple)) else [logits_or_list]

    if(len(heads) == 1):
        # no deep supervision, just compute loss on the single head
        return loss_fn(heads[0], label)
    

    assert len(heads) == len(level_weights), (
        f"model returned {len(heads)} DS heads, expected {len(level_weights)}"
    )

    total = 0.0
    for w, logits in zip(level_weights, heads):
        if debug:
            print(
                f"  [DEBUG_DS] head logits={tuple(logits.shape)} "
                f"label={tuple(label.shape)}"
            )
        tgt = _resize_label_to_logits(label, logits.shape[-3:], debug=debug)
        total = total + w * loss_fn(logits, tgt)
    return total / max(sum(level_weights), 1e-8)


# ══════════════════════════════════════════════════════════════════
#  Training / validation loops
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:        nn.Module,
    loader,
    optimizer,
    scaler:       GradScaler,
    loss_fn:      nn.Module,
    device:       torch.device,
    epoch:        int,
    cfg_train:    dict,
    run_paths:    Dict[str, Path],
    logger:       logging.Logger,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    reset_all_save_counters(model)

    # ── Resolve per-epoch dump policy from config ─────────────
    # `DUMP_LAST_STEP_ONLY` wins over `DUMP_FIRST_STEP_ONLY` when both
    # are set, so flipping the new flag takes effect without touching
    # the old one. The third state — both False — means "dump every
    # step" (the modules write `step_{N}_*.npy` each forward).
    dump_last_only  = bool(cfg_train.get("DUMP_LAST_STEP_ONLY", False))
    dump_first_only = bool(cfg_train.get("DUMP_FIRST_STEP_ONLY", False)) and not dump_last_only
    if dump_last_only:
        dump_mode = "last_step"
    elif dump_first_only:
        dump_mode = "first_step"
    else:
        dump_mode = "all_steps"

    # Tag every DA + FNO module with this epoch + the chosen mode so
    # the on-disk filename embeds the epoch id when applicable
    # (e.g. epoch_007_last_step_*.npy in last_step mode).
    _set_epoch_on_modules(model, epoch_id=epoch, dump_mode=dump_mode)

    grad_clip = float(cfg_train.get("GRAD_CLIP_NORM", 1.0))

    losses = deque(maxlen=64)
    t0 = time.time()

    # Resolve AMP dtype + optional input cast from config (single source of
    # truth — also used by validate()).
    use_amp, amp_dtype, force_in_dtype = _resolve_amp_dtypes(cfg_train)

    # Number of steps this epoch — needed so last_step mode can clear
    # save_dirs on every intermediate step and only restore them on
    # the very last one. Saves the splat/np.save cost on N-1 steps.
    try:
        n_steps = len(loader)
    except TypeError:
        n_steps = None   # loader isn't sized (custom iterable); fall back to "write every step"
    dumped_first = False

    for step_idx, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True).to(force_in_dtype)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(x)
            loss = ds_loss(
                out, y, loss_fn, cfg_train["DS_LEVEL_WEIGHTS"],
                debug=bool(cfg_train.get("DEBUG_DS", False)),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))

        # FREE MEMORY IMMEDIATELY AFTER STEP
        del x, y, out, loss
        torch.cuda.empty_cache()
        gc.collect()

        # ── Per-epoch dump policy ────────────────────────────
        # first_step mode (legacy): dump on step 0 only, then disable saves
        #   for the rest of the epoch.
        # last_step mode:  write ONLY on the final step (saves N-1 wasted
        #   writes/splats); the filename is `epoch_{N:03d}_last_step_*`.
        # all_steps mode:  leave save_dirs ON every step; module filenames
        #   are `step_{N}_*.npy`.
        is_last_step = (n_steps is None) or (step_idx == n_steps - 1)
        if dump_mode == "first_step":
            if not dumped_first:
                dumped_first = True
                _set_save_dirs(model, None)            # disable after first write
        elif dump_mode == "last_step":
            if not is_last_step and n_steps is not None:
                _set_save_dirs(model, None)            # suppress intermediate writes
            elif is_last_step and n_steps is not None:
                _restore_save_dirs(model, run_paths)   # re-enable just for the final write
        # else (all_steps): save_dirs stay ON the whole epoch.

    # Final hygiene: if we suppressed intermediate writes in last_step
    # mode but never restored (e.g. loader was empty / unknown length),
    # make sure save_dirs are back ON for the next epoch's first step.
    if dump_mode == "last_step":
        _restore_save_dirs(model, run_paths)

    dt = time.time() - t0
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    logger.info(
        f"  epoch {epoch:03d} | train loss {mean_loss:.4f} | "
        f"steps {len(loader)} | {dt:.1f}s"
    )
    return {"train_loss": mean_loss, "epoch_time_s": dt}


import gc
import torch
import numpy as np
from typing import Dict

def _resolve_amp_dtypes(cfg_train: dict) -> tuple[bool, torch.dtype, torch.dtype]:
    """Resolve (use_amp, autocast dtype, forced-input dtype) from config.

    `AMP_DTYPE` controls the autocast dtype ("float16" or "bfloat16").
    `FORCE_INPUT_DTYPE` controls the explicit cast applied to the input
    tensor *before* the model (None, "float32", "float16", "bfloat16").
    When `FORCE_INPUT_DTYPE` is None we default to fp16 if AMP is on,
    else fp32.
    """
    use_amp = bool(cfg_train.get("AMP", True)) and torch.cuda.is_available()
    amp_dtype_str = str(cfg_train.get("AMP_DTYPE", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16

    force_in_dtype = cfg_train.get("FORCE_INPUT_DTYPE", None)
    if force_in_dtype is not None:
        force_in_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(str(force_in_dtype).lower())
    if force_in_dtype is None:
        force_in_dtype = torch.float16 if use_amp else torch.float32
    return use_amp, amp_dtype, force_in_dtype


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    cfg_train: dict,
) -> Dict[str, float]:
    """Validation pass with memory-optimized sliding-window inference.

    Honors the same `AMP` / `AMP_DTYPE` / `FORCE_INPUT_DTYPE` knobs from
    `config.py` that `train_one_epoch` does, so validation VRAM matches
    training VRAM.
    """
    model.eval()
    roi_size = tuple(cfg.dataloader_config["ROI_SIZE"])
    sw_batch_size = 1
    sw_overlap = 0.5

    use_amp, amp_dtype, force_in_dtype = _resolve_amp_dtypes(cfg_train)

    losses = []
    dices = []

    for batch in loader:
        # Match training-loop behaviour: cast input to the configured dtype
        # *before* the model sees it, so fp16 config really runs fp16.
        x = batch["image"].to(device, non_blocking=True).to(force_in_dtype)
        y = batch["label"].to(device, non_blocking=True)

        # Slide inference over patch-sized windows — wrapped in autocast so
        # the model itself runs in the configured dtype, not float32.
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = sliding_window_inference(
                inputs=x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=sw_overlap,
                predictor=model,
            )

        # Unpack deep-supervision output if returning a collection
        if isinstance(out, (list, tuple)):
            out = out[0]

        # Compute validation loss (autocast is still active here, so loss
        # runs in the same dtype as the predictions — matches training).
        val_loss = loss_fn(out, y).item()
        losses.append(float(val_loss))

        # Binary per-volume Dice without allocating intermediate tensor copies
        pred_mask = out.argmax(dim=1, keepdim=True) > 0
        target_mask = y > 0

        intersection = (pred_mask & target_mask).sum().item()
        total = pred_mask.sum().item() + target_mask.sum().item()

        dice = (2.0 * intersection / total) if total > 0 else 1.0
        dices.append(dice)

        # Free volume references immediately to prevent Host/Device OOM spikes
        del x, y, out, pred_mask, target_mask

    # Force cleanup after validation loop
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_dice": float(np.mean(dices)) if dices else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════
#  Save-dir toggling (skip dumps for steps after the first one)
# ══════════════════════════════════════════════════════════════════

# We snapshot the original save_dirs at construction time so we can
# safely toggle them off mid-epoch and back on at the next epoch start.
_ORIG_SAVE_DIRS: Dict[int, Optional[str]] = {}


def _snapshot_save_dirs(model: nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "save_dir"):
            _ORIG_SAVE_DIRS[id(m)] = m.save_dir


def _set_save_dirs(model: nn.Module, value: Optional[str]) -> None:
    for m in model.modules():
        if hasattr(m, "save_dir"):
            m.save_dir = value


def _restore_save_dirs(model: nn.Module, run_paths: Dict[str, Path]) -> None:
    """Re-establish the per-level folders for the NEXT epoch's first step."""
    base = run_paths["run"]  # the run dir
    for m in model.modules():
        mid = id(m)
        if mid in _ORIG_SAVE_DIRS and _ORIG_SAVE_DIRS[mid] is not None:
            m.save_dir = _ORIG_SAVE_DIRS[mid]
        elif hasattr(m, "save_dir") and m.save_dir is None:
            # leave as None — fine
            pass


def _set_epoch_on_modules(model: nn.Module, epoch_id: int, dump_mode: str) -> None:
    """Stamp `epoch_id` + `dump_mode` on every DA / FNO module.

    Both `DeformableAttention3d` and `SpectralConv3d` expose `set_epoch(...)`;
    the on-disk filename pattern (e.g. `epoch_{N:03d}_last_step_*.npy`) is
    derived from these two fields inside each module's forward.
    """
    for m in model.modules():
        if hasattr(m, "set_epoch") and callable(getattr(m, "set_epoch")):
            try:
                m.set_epoch(epoch_id, dump_mode)
            except Exception:
                # never let bookkeeping crash training
                pass


# ══════════════════════════════════════════════════════════════════
#  Checkpointing
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:      nn.Module,
    optimizer,
    scaler:     GradScaler,
    epoch:      int,
    metrics:    dict,
    run_paths:  Dict[str, Path],
    is_best:    bool,
) -> Path:
    state = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "epoch":     epoch,
        "metrics":   metrics,
        "model_config": cfg.model_config,
        "train_config": cfg.train_config,
    }

    policy = str(cfg.train_config.get("CKPT_POLICY", "best")).lower()
    if policy not in ("best", "last", "both"):
        policy = "best"

    last_path = run_paths["ckpts"] / "last.pth"
    best_path = run_paths["ckpts"] / "best_dice.pth"

    # Write `last.pth` if policy asks for it.
    if policy in ("last", "both"):
        torch.save(state, last_path)

    # Write `best_dice.pth` if it's the new best (always for "best"/"both";
    # also for "last" so --resume still finds something at startup).
    if is_best and policy in ("best", "both"):
        torch.save(state, best_path)
    elif is_best and policy == "last":
        # No `last` was kept (since this *is* the last). Mirror to best so
        # resume still works, but the on-disk file we return is `last`.
        torch.save(state, best_path)
        torch.save(state, last_path)

    # Per-epoch checkpoints (only when policy wants them OR the legacy flag is on).
    save_every = bool(cfg.train_config.get("SAVE_EVERY_EPOCH", False))
    if save_every and policy == "both":
        epoch_path = run_paths["ckpts"] / f"epoch_{epoch:03d}.pth"
        torch.save(state, epoch_path)
        keep_n = int(cfg.train_config.get("KEEP_LAST_N_CKPTS", 3))
        if keep_n > 0:
            per_epoch = sorted(run_paths["ckpts"].glob("epoch_*.pth"))
            for old in per_epoch[:-keep_n]:
                try:
                    old.unlink()
                except OSError:
                    pass

    # Final hygiene: under "best", a stale `last.pth` from a previous run
    # would waste space — remove it if the policy doesn't allow it.
    if policy == "best" and last_path.exists() and not is_best:
        try:
            last_path.unlink()
        except OSError:
            pass

    # Return the path the caller should treat as "the latest".
    if policy == "best":
        return best_path if best_path.exists() else last_path
    return last_path


def load_checkpoint(path: Path, model, optimizer=None, scaler=None) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("epoch", -1))


# ══════════════════════════════════════════════════════════════════
#  Smoke-test data slicing (limit to a few samples, no caching thrash)
# ══════════════════════════════════════════════════════════════════

def _shrink_dataset_for_smoke(loader, max_samples: int):
    """Replace a DataLoader's underlying dataset with a tiny subset."""
    base_ds = loader.dataset
    if hasattr(base_ds, "data"):
        base_ds.data = base_ds.data[:max_samples]
    # Wrap into a fresh DataLoader with the same constructor kwargs
    from torch.utils.data import DataLoader
    return DataLoader(
        base_ds,
        batch_size=loader.batch_size,
        shuffle=False,        # deterministic for smoke test
        num_workers=0,
        sampler=None,
        collate_fn=loader.collate_fn,
    )


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_name", help="Run name; creates runs/<name>/")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Tiny 2-epoch run on a few samples; overrides EPOCHS.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from runs/<name>/ckpts/last.pth if present.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_paths = setup_run_dirs(args.experiment_name)
    save_config_snapshot(run_paths["run"])
    logger = get_logger(run_paths["logs"] / "train.log")
    logger.info(f"=== train.py  |  experiment: {args.experiment_name}  |  device: {device} ===")
    logger.info(f"run dir: {run_paths['run']}")

    # ── DataLoaders ──────────────────────────────────────────
    logger.info("building dataloaders…")
    train_loader, val_loader, test_loader = build_dataloaders(str(SPLITS_JSON))

    _v = bool(cfg.model_config.get("VERBOSE", False))
    if _v:
        print("[verbose] train batch size:", train_loader.batch_size)
        print("[verbose] dataloader config:", cfg.dataloader_config)

    # ── Auto-detect in_channels from the first batch ─────────
    cfg.model_config["IN_CHANNELS"] = auto_detect_in_channels(train_loader)
    logger.info(f"auto-detected IN_CHANNELS = {cfg.model_config['IN_CHANNELS']}")

    # ── Smoke-test overrides ─────────────────────────────────
    cfg_train = dict(cfg.train_config)  # shallow copy
    if args.smoke_test:
        cfg_train["EPOCHS"]            = 5
        cfg_train["SAVE_EVERY_EPOCH"]  = True
        cfg_train["KEEP_LAST_N_CKPTS"] = 1
        logger.info("SMOKE-TEST mode: 2 epochs, tiny subset")
        train_loader = _shrink_dataset_for_smoke(train_loader, max_samples=2)
        val_loader   = _shrink_dataset_for_smoke(val_loader,   max_samples=1)

    # ── Model ────────────────────────────────────────────────
    dump_dir = run_paths["run"] if (cfg_train.get("DUMP_ATTN") or cfg_train.get("DUMP_FNO")) else None
    model = RIUnet(
        in_channels=cfg.model_config["IN_CHANNELS"],
        save_dir=str(dump_dir) if dump_dir else None,
    ).to(device)

    # Cast parameters (weights + biases) to the dtype requested by config.
    # `None` or "float32" leaves them in fp32 (GradScaler needs fp32
    # master weights for safe fp16 training). "float16" / "bfloat16"
    # actually moves parameter memory to that dtype — biggest VRAM win
    # when paired with the same AMP_DTYPE.
    weights_dtype_str = cfg_train.get("WEIGHTS_DTYPE", None)
    weights_dtype = None
    if weights_dtype_str is not None:
        weights_dtype = {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(str(weights_dtype_str).lower())
        if weights_dtype is None:
            raise ValueError(
                f"Invalid WEIGHTS_DTYPE={weights_dtype_str!r}; "
                f"expected None, 'float32', 'float16', or 'bfloat16'."
            )
        if weights_dtype != torch.float32:
            model = model.to(dtype=weights_dtype)
            logger.info(
                f"cast model parameters to {weights_dtype} "
                f"(per train_config['WEIGHTS_DTYPE'])"
            )

    # Guard: GradScaler only works with fp16 autocast over fp32 master
    # weights. If the user asked for both fp16 AMP *and* fp16 weights,
    # we must disable GradScaler or it will error out at the first step.
    amp_enabled    = bool(cfg_train.get("AMP", True)) and device.type == "cuda"
    amp_dtype_str  = str(cfg_train.get("AMP_DTYPE", "float16")).lower()
    amp_is_fp16    = (amp_dtype_str == "float16")
    scaler_enabled = amp_enabled and not (amp_is_fp16 and weights_dtype == torch.float16)
    if amp_enabled and amp_is_fp16 and weights_dtype == torch.float16:
        logger.warning(
            "WEIGHTS_DTYPE='float16' + AMP_DTYPE='float16' is incompatible "
            "with GradScaler; disabling scaler (loss scaling). Consider "
            "WEIGHTS_DTYPE=None or 'float32' for true mixed-precision."
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"RIUnet built: {n_params/1e6:.2f}M trainable params, save_dir={dump_dir}")
    _v = bool(cfg.model_config.get("VERBOSE", False))
    if _v:
        print("[verbose] dump_dir:", dump_dir, "DUMP_ATTN:", cfg_train.get("DUMP_ATTN"),
              "DUMP_FNO:", cfg_train.get("DUMP_FNO"))
    _snapshot_save_dirs(model)

    # ── Loss / optimizer / scheduler / AMP ───────────────────
    loss_fn = _build_loss(cfg_train)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg_train["LR"]),
        weight_decay=float(cfg_train["WEIGHT_DECAY"]),
    )
    # scaler = GradScaler(enabled=(device.type == "cuda" and bool(cfg_train.get("AMP", True))))
    # In main():
    scaler = GradScaler("cuda", enabled=scaler_enabled)

    # ── VERBOSE: dtype audit ─────────────────────────────────
    # Print exactly which dtype is being used for every tensor that
    # matters at training time. Runs once at startup, gated by either
    # `model_config["VERBOSE"]` or the new `train_config["VERBOSE_DTYPE"]`
    # so it can be toggled independently of model-internal prints.
    _verbose_dtype = bool(
        cfg_train.get("VERBOSE_DTYPE",
                      cfg.model_config.get("VERBOSE", False))
    )
    if _verbose_dtype:
        # Resolve the same values the loops will use
        _use_amp, _amp_dtype, _force_in_dtype = _resolve_amp_dtypes(cfg_train)

        # Spot-check actual parameter dtypes (after the .to(dtype=...)
        # cast above) so the report matches reality, not just config.
        param_dtypes = {p.dtype for p in model.parameters()}
        if len(param_dtypes) == 1:
            actual_weight_dtype = next(iter(param_dtypes))
        else:
            actual_weight_dtype = param_dtypes  # mixed (rare)

        # Sample one activation dtype by running a single forward pass on
        # a tiny synthetic batch — confirms autocast is actually doing
        # what we think it is. Wrapped in autocast so the dtype we see
        # is the runtime dtype, not the parameter dtype.
        try:
            _probe_shape = (1, cfg.model_config["IN_CHANNELS"],
                            cfg.dataloader_config["ROI_SIZE"][0] // 4,
                            cfg.dataloader_config["ROI_SIZE"][1] // 4,
                            cfg.dataloader_config["ROI_SIZE"][2] // 4)
            with torch.no_grad():
                _probe_x = torch.zeros(_probe_shape, device=device).to(_force_in_dtype)
                with torch.amp.autocast("cuda", enabled=_use_amp, dtype=_amp_dtype):
                    _probe_out = model(_probe_x)
            if isinstance(_probe_out, (list, tuple)):
                _probe_out = _probe_out[0]
            activation_dtype = _probe_out.dtype
            del _probe_x, _probe_out
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as _e:
            activation_dtype = f"<probe failed: {_e.__class__.__name__}>"

        # Optimizer-state dtype (always fp32 — that's how PyTorch stores it)
        opt_state_dtypes = set()
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    opt_state_dtypes.add(v.dtype)
        opt_state_dtype = (next(iter(opt_state_dtypes))
                           if len(opt_state_dtypes) == 1
                           else opt_state_dtypes)

        # Scaler status
        scaler_status = (
            f"enabled (init_scale={scaler.get_scale():.0f})"
            if scaler_enabled else "disabled"
        )

        # ── Print the report ──────────────────────────────────
        sep = "─" * 60
        logger.info(sep)
        logger.info(" DTYPE AUDIT  (single source of truth = config.py)")
        logger.info(sep)
        logger.info(f"  input tensor   : {_force_in_dtype}   "
                    f"(train_config['FORCE_INPUT_DTYPE'])")
        logger.info(f"  autocast       : dtype={_amp_dtype}, "
                    f"enabled={_use_amp}   "
                    f"(train_config['AMP_DTYPE'] / 'AMP')")
        logger.info(f"  model weights  : {actual_weight_dtype}   "
                    f"(train_config['WEIGHTS_DTYPE'])")
        logger.info(f"  forward output : {activation_dtype}   "
                    f"(measured via probe forward pass)")
        logger.info(f"  optimizer state: {opt_state_dtype}   "
                    f"(always fp32 in PyTorch)")
        logger.info(f"  grad scaler    : {scaler_status}")
        logger.info(f"  device         : {device}")
        logger.info(sep)

    epochs        = int(cfg_train["EPOCHS"])
    warmup_epochs = int(cfg_train.get("WARMUP_EPOCHS", 0))

    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda e: min((e + 1) / warmup_epochs, 1.0),
        )
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    # ── Optional resume ─────────────────────────────────────
    start_epoch   = 0
    best_val_dice = -1.0
    bad_epochs    = 0
    if args.resume:
        last = run_paths["ckpts"] / "last.pth"
        if last.exists():
            start_epoch = load_checkpoint(last, model, optimizer, scaler) + 1
            logger.info(f"resumed from {last} at epoch {start_epoch}")
            best_ckpt = run_paths["ckpts"] / "best_dice.pth"
            if best_ckpt.exists():
                best_val_dice = float(
                    torch.load(best_ckpt, map_location="cpu", weights_only=False)
                    .get("metrics", {}).get("val_dice", -1.0)
                )

    # ── Training loop ───────────────────────────────────────
    history = []
    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device,
            epoch, cfg_train, run_paths, logger,
        )
        scheduler.step()

        val_stats = validate(model, val_loader, loss_fn, device, cfg_train)
        logger.info(
            f"  epoch {epoch:03d} | val loss {val_stats['val_loss']:.4f}  "
            f"val dice {val_stats['val_dice']:.4f}  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        history.append({"epoch": epoch, **train_stats, **val_stats})

        is_best = val_stats["val_dice"] > best_val_dice
        if is_best:
            best_val_dice = val_stats["val_dice"]
            bad_epochs = 0
        else:
            bad_epochs += 1

        save_checkpoint(
            model, optimizer, scaler, epoch,
            metrics={**train_stats, **val_stats, "best_val_dice": best_val_dice},
            run_paths=run_paths,
            is_best=is_best,
        )

        # Early stop
        patience = int(cfg_train.get("EARLY_STOP_PATIENCE", 0))
        if patience > 0 and bad_epochs >= patience:
            logger.info(f"early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Persist history
    with open(run_paths["run"] / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)

    logger.info(f"=== done.  best val_dice = {best_val_dice:.4f} ===")


def _build_loss(cfg_train: dict) -> nn.Module:
    """Build a DiceCELoss per train_config. Falls back to plain CE if MONAI missing."""
    try:
        from monai.losses import DiceCELoss
        print("Using MONAI DiceCELoss (CE + Dice) as the loss function. You can change weights in config.py.")
        return DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            weight=None,
            sigmoid=False,
            lambda_dice=float(cfg_train.get("DICE_WEIGHT", 0.5)),
            lambda_ce=float(cfg_train.get("CE_WEIGHT", 0.5)),
        )
    except Exception as actaul_error:
        # Fallback: plain CE
        if(actaul_error.__class__.__name__ == "ModuleNotFoundError"):
            print("MONAI not found. Install MONAI for DiceCELoss.")
        else:
            raise Exception("MONAI not found; please install monai to use DiceCELoss. Actual Error: " + str(actaul_error))
        


if __name__ == "__main__":
    main()
