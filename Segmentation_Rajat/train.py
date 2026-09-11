"""
train.py — Train RIUnet (CAC segmentation), single-channel sigmoid output.

CLI:
    python Segmentation_Rajat/train.py EXPERIMENT_NAME
    python Segmentation_Rajat/train.py EXPERIMENT_NAME --smoke-test
    python Segmentation_Rajat/train.py EXPERIMENT_NAME --resume

Confirmed via overfit_check.py / model_sanity_check.py:
    - RIUnet outputs 1 channel (raw logits, no internal activation).
    - Plain unweighted DiceCE / softmax-based losses FAIL on this data's
      sparsity (~0.001-0.005% foreground) — collapses to all-background.
    - pos_weight-based sigmoid Dice+BCE or Tversky+BCE PASS.
    This file only uses the confirmed-working (sigmoid, pos_weight) path.

SMOKE-TEST NOTE:
    --smoke-test now live-samples SMOKE_N_SAMPLES (default 10) fresh
    batches per epoch directly from the full train_loader — a re-pull each
    epoch, same as overfit_check.py's --sample-mode live — instead of
    physically shrinking the dataset to a static first-N subset. This is
    the sampling behavior that was confirmed to converge in overfit_check.
    AMP and the GradScaler are controlled entirely by config.py
    (train_config["AMP"], and the scaler is always constructed with
    enabled=False below) — set AMP: False in config to match the
    overfit_check.py run, which never used autocast.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, TverskyLoss
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from monai.inferers import sliding_window_inference

import config as cfg
from dataset import build_dataloaders
from model import RIUnet

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
SPLITS_JSON = Path(cfg.preprocessing_config["SPLITS_JSON"])


# ══════════════════════════════════════════════════════════════════
#  Paths / logging / seed
# ══════════════════════════════════════════════════════════════════

def setup_run_dirs(experiment_name: str) -> Dict[str, Path]:
    run_dir = RUNS_DIR / experiment_name
    paths = {
        "run": run_dir, "ckpts": run_dir / "ckpts", "logs": run_dir / "logs",
        "attn": run_dir / "attn_maps", "fno": run_dir / "fno_maps",
        "eval": run_dir / "eval_preds", "snapshot": run_dir / "config_snapshot.json",
    }
    for k, p in paths.items():
        if k != "snapshot":
            p.mkdir(parents=True, exist_ok=True)
    return paths


def save_config_snapshot(run_dir: Path, extra: dict | None = None) -> None:
    snap = {
        "model_config": cfg.model_config,
        "dataloader_config": cfg.dataloader_config,
        "train_config": cfg.train_config,
        "hu_config": cfg.HU_CONFIG,
    }
    if extra:
        snap["runtime"] = extra
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
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_detect_in_channels(loader) -> int:
    return int(next(iter(loader))["image"].shape[1])


def reset_all_save_counters(model: nn.Module) -> None:
    for m in model.modules():
        if hasattr(m, "reset_save_counter"):
            m.reset_save_counter()


# ══════════════════════════════════════════════════════════════════
#  Loss — single-channel (sigmoid) only, pos_weight-based
# ══════════════════════════════════════════════════════════════════

class WeightedDiceCE(nn.Module):
    """Dice(sigmoid) + pos_weight-BCE. Confirmed to converge on RIUnet."""

    def __init__(self, pos_weight: float, lambda_dice: float = 0.5, lambda_ce: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.register_buffer("pos_weight_t", torch.tensor(pos_weight))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_t)
        self.lambda_dice, self.lambda_ce = lambda_dice, lambda_ce

    def forward(self, pred, target):
        target = target.float()
        return self.lambda_dice * self.dice(pred, target) + self.lambda_ce * self.bce(pred, target)


class TverskyWeightedCE(nn.Module):
    """Tversky(sigmoid) + pos_weight-BCE. Converged fastest in sanity checks."""

    def __init__(self, pos_weight: float, alpha: float = 0.3, beta: float = 0.7,
                 lambda_tversky: float = 0.5, lambda_ce: float = 0.5):
        super().__init__()
        self.tversky = TverskyLoss(sigmoid=True, alpha=alpha, beta=beta)
        self.register_buffer("pos_weight_t", torch.tensor(pos_weight))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_t)
        self.lambda_tversky, self.lambda_ce = lambda_tversky, lambda_ce

    def forward(self, pred, target):
        target = target.float()
        return self.lambda_tversky * self.tversky(pred, target) + self.lambda_ce * self.bce(pred, target)


def estimate_pos_weight(loader, max_batches: int, cap: float) -> float:
    """bg:fg voxel ratio over a sample of training batches, capped to avoid
    instability. Real ratios here run ~30,000:1 to 90,000:1."""
    total_fg = total_vox = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        y = batch["label"]
        total_fg += int((y > 0).sum().item())
        total_vox += int(y.numel())
    if total_fg == 0:
        return cap
    return float(min((total_vox - total_fg) / total_fg, cap))


def _build_loss(cfg_train: dict, pos_weight: float) -> nn.Module:
    loss_type = str(cfg_train.get("LOSS_TYPE", "DICECE")).upper()
    lambda_loss = float(cfg_train.get("DICE_WEIGHT", cfg_train.get("TVERSKY_WEIGHT", 0.5)))
    lambda_ce = float(cfg_train.get("CE_WEIGHT", 0.5))

    if loss_type == "TVERSKYCE":
        alpha = float(cfg_train.get("TVERSKY_ALPHA", 0.3))
        beta = float(cfg_train.get("TVERSKY_BETA", 0.7))
        return TverskyWeightedCE(pos_weight, alpha, beta, lambda_loss, lambda_ce)
    if loss_type == "DICECE":
        return WeightedDiceCE(pos_weight, lambda_loss, lambda_ce)
    raise ValueError(f"Unsupported LOSS_TYPE: {loss_type}. Use 'DiceCE' or 'TverskyCE'.")


# ══════════════════════════════════════════════════════════════════
#  DS loss helper
# ══════════════════════════════════════════════════════════════════

def _resize_label_to_logits(label: torch.Tensor, target_hw: tuple) -> torch.Tensor:
    if label.shape[-3:] == target_hw:
        return label
    return F.interpolate(label.float(), size=target_hw, mode="nearest")


def ds_loss(logits_or_list, label, loss_fn, level_weights: List[float]) -> torch.Tensor:
    heads = logits_or_list if isinstance(logits_or_list, (list, tuple)) else [logits_or_list]
    if len(heads) == 1:
        return loss_fn(heads[0], label)
    assert len(heads) == len(level_weights), (
        f"model returned {len(heads)} DS heads, expected {len(level_weights)}"
    )
    total = 0.0
    for w, logits in zip(level_weights, heads):
        total = total + w * loss_fn(logits, _resize_label_to_logits(label, logits.shape[-3:]))
    return total / max(sum(level_weights), 1e-8)


# ══════════════════════════════════════════════════════════════════
#  Train / validate
# ══════════════════════════════════════════════════════════════════

def _resolve_amp_dtypes(cfg_train: dict):
    use_amp = bool(cfg_train.get("AMP", True)) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if str(cfg_train.get("AMP_DTYPE", "float16")).lower() == "bfloat16" else torch.float16
    force_in = cfg_train.get("FORCE_INPUT_DTYPE", None)
    force_in = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        str(force_in).lower()) if force_in else (torch.float16 if use_amp else torch.float32)
    return use_amp, amp_dtype, force_in


def _live_subset_iter(loader, n_samples: int):
    """Pull only the first n_samples batches out of `loader`, fresh each
    call. Mirrors overfit_check.py's --sample-mode live: since this
    re-iterates the loader from the start each epoch, a shuffling loader
    yields a different random n_samples-sized subset every epoch."""
    for step, batch in enumerate(loader):
        if step >= n_samples:
            break
        yield batch


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, epoch, cfg_train, run_paths, logger,
                     n_samples: int | None = None):
    model.train()
    reset_all_save_counters(model)
    dump_epochs = cfg_train.get("DUMP_EPOCHS", [])
    should_dump = epoch in dump_epochs
    grad_clip = float(cfg_train.get("GRAD_CLIP_NORM", 1.0))
    losses = deque(maxlen=64)
    t0 = time.time()
    use_amp, amp_dtype, force_in_dtype = _resolve_amp_dtypes(cfg_train)

    # --sample-mode live: only pull the first n_samples batches this epoch,
    # instead of iterating the full loader (see _live_subset_iter above).
    if n_samples is not None:
        step_source = _live_subset_iter(loader, n_samples)
        n_steps = n_samples
    else:
        step_source = loader
        n_steps = len(loader) if hasattr(loader, "__len__") else None

    @torch.no_grad()
    def debug_logit_stats(out: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Sigmoid probability stats, split by whether the voxel is foreground
        or background in the label.
        """
        # Pick the highest-resolution tensor if deep supervision is enabled
        logits = out[0] if isinstance(out, (list, tuple)) else out
        probs = torch.sigmoid(logits)
        
        fg_mask, bg_mask = y > 0, y <= 0
        fg_mean = probs[fg_mask].mean().item() if fg_mask.any() else float("nan")
        bg_mean = probs[bg_mask].mean().item() if bg_mask.any() else float("nan")
        
        logger.info(
            f"    [train probs] min={probs.min().item():.4f} mean={probs.mean().item():.4f} "
            f"max={probs.max().item():.4f} | mean_prob@fg={fg_mean:.4f} mean_prob@bg={bg_mean:.4f}"
        )
        return {"fg_mean": fg_mean, "bg_mean": bg_mean}

    logit_stats = {}
    steps_run = 0

    for step_idx, batch in enumerate(step_source):
        x = batch["image"].to(device, non_blocking=True).to(force_in_dtype)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        is_last = (n_steps is None) or (step_idx == n_steps - 1)
        dump_dir = str(run_paths["run"] / f"epoch_{epoch:03d}") if (should_dump and is_last) else None

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            out = model(x, da_save_dir=dump_dir)
            loss = ds_loss(out, y, loss_fn, cfg_train["DS_LEVEL_WEIGHTS"])

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(float(loss.item()))
        steps_run += 1

        # Compute logit stats on the final step before clearing tensors
        if is_last:
            logit_stats = debug_logit_stats(out, y)

        del x, y, out, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    dt = time.time() - t0
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    logger.info(f"  epoch {epoch:03d} | train loss {mean_loss:.4f} | steps {steps_run} | {dt:.1f}s | dump={should_dump}")
    
    return {
        "train_loss": mean_loss, 
        "epoch_time_s": dt,
        "train_fg_mean_prob": logit_stats.get("fg_mean", float("nan")),
        "train_bg_mean_prob": logit_stats.get("bg_mean", float("nan")),
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg_train, logger=None) -> Dict[str, float]:
    """Dice via sigmoid+threshold — NOT argmax (out has 1 channel; argmax
    over a size-1 dim is always 0, which silently zeroes dice forever)."""
    model.eval()
    roi_size = tuple(cfg.dataloader_config["ROI_SIZE"])
    sw_batch_size = int(cfg_train.get("VAL_SW_BATCH_SIZE", 4))
    sw_overlap = float(cfg_train.get("VAL_SW_OVERLAP", 0.25))
    threshold = float(cfg_train.get("VAL_THRESHOLD", 0.5))
    losses, dices = [], []

    def debug_logit_stats(out: torch.Tensor, y: torch.Tensor) -> dict:
        """Sigmoid probability stats, split by foreground vs background."""
        probs = torch.sigmoid(out)
        fg_mask, bg_mask = y > 0, y <= 0
        fg_mean = probs[fg_mask].mean().item() if fg_mask.any() else float("nan")
        bg_mean = probs[bg_mask].mean().item() if bg_mask.any() else float("nan")
        
        if logger:
            logger.info(
                f"    [val probs] min={probs.min().item():.4f} mean={probs.mean().item():.4f} "
                f"max={probs.max().item():.4f} | mean_prob@fg={fg_mean:.4f} mean_prob@bg={bg_mean:.4f}"
            )
        return {"fg_mean": fg_mean, "bg_mean": bg_mean}

    logit_stats = {}
    n_steps = len(loader) if hasattr(loader, "__len__") else None

    for step_idx, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True).to(torch.float32)
        y = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32):
            out = sliding_window_inference(
                inputs=x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                overlap=sw_overlap, predictor=model, mode="gaussian", device=device,
            )
        if isinstance(out, (list, tuple)):
            out = out[0]

        losses.append(loss_fn(out, y).detach())

        pred_mask = torch.sigmoid(out) > threshold
        target_mask = y > 0
        intersection = (pred_mask & target_mask).sum()
        total = pred_mask.sum() + target_mask.sum()
        dices.append(torch.where(total > 0, (2.0 * intersection) / total, torch.tensor(1.0, device=device)))

        # Log probability stats on the final validation volume
        is_last = (n_steps is None) or (step_idx == n_steps - 1)
        if is_last:
            logit_stats = debug_logit_stats(out, y)

    mean_loss = torch.stack(losses).mean().item() if losses else float("nan")
    mean_dice = torch.stack(dices).mean().item() if dices else float("nan")
    
    return {
        "val_loss": float(mean_loss),
        "val_dice": float(mean_dice),
        "val_fg_mean_prob": logit_stats.get("fg_mean", float("nan")),
        "val_bg_mean_prob": logit_stats.get("bg_mean", float("nan")),
    }

# ══════════════════════════════════════════════════════════════════
#  Checkpointing
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scaler, epoch, metrics, run_paths, is_best) -> Path:
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "metrics": metrics,
        "model_config": cfg.model_config, "train_config": cfg.train_config,
    }
    policy = str(cfg.train_config.get("CKPT_POLICY", "best")).lower()
    if policy not in ("best", "last", "both"):
        policy = "best"
    last_path = run_paths["ckpts"] / "last.pth"
    best_path = run_paths["ckpts"] / "best_dice.pth"

    if policy in ("last", "both"):
        torch.save(state, last_path)
    if is_best and policy in ("best", "both"):
        torch.save(state, best_path)
    elif is_best and policy == "last":
        torch.save(state, best_path); torch.save(state, last_path)

    if policy == "both" and bool(cfg.train_config.get("SAVE_EVERY_EPOCH", False)):
        epoch_path = run_paths["ckpts"] / f"epoch_{epoch:03d}.pth"
        torch.save(state, epoch_path)
        keep_n = int(cfg.train_config.get("KEEP_LAST_N_CKPTS", 3))
        if keep_n > 0:
            per_epoch = sorted(run_paths["ckpts"].glob("epoch_*.pth"))
            for old in per_epoch[:-keep_n]:
                try: old.unlink()
                except OSError: pass

    if policy == "best" and last_path.exists() and not is_best:
        try: last_path.unlink()
        except OSError: pass

    return best_path if policy == "best" and best_path.exists() else last_path


def load_checkpoint(path: Path, model, optimizer=None, scaler=None) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("epoch", -1))


def _shrink_dataset_for_smoke(loader, max_samples: int):
    base_ds = loader.dataset
    if hasattr(base_ds, "data"):
        base_ds.data = base_ds.data[:max_samples]
    from torch.utils.data import DataLoader
    return DataLoader(base_ds, batch_size=loader.batch_size, shuffle=False,
                       num_workers=0, sampler=None, collate_fn=loader.collate_fn)



# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_name")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-n-samples", type=int, default=10,
                     help="Live batches/epoch during --smoke-test (fresh pull from the full "
                          "train_loader each epoch, matching overfit_check.py's "
                          "--sample-mode live). Does not affect normal (non-smoke-test) runs.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_paths = setup_run_dirs(args.experiment_name)
    logger = get_logger(run_paths["logs"] / "train.log")
    logger.info(f"=== train.py | experiment: {args.experiment_name} | device: {device} ===")

    logger.info("building dataloaders…")
    train_loader, val_loader, _ = build_dataloaders(str(SPLITS_JSON))
    cfg.model_config["IN_CHANNELS"] = auto_detect_in_channels(train_loader)
    logger.info(f"auto-detected IN_CHANNELS = {cfg.model_config['IN_CHANNELS']}")

    cfg_train = dict(cfg.train_config)
    proc = psutil.Process()

    smoke_n_samples = None
    if args.smoke_test:
        cfg_train["EPOCHS"] = 100
        smoke_n_samples = args.smoke_n_samples
        logger.info(f"SMOKE-TEST mode: live-sampling {smoke_n_samples} fresh batch(es)/epoch "
                    f"from train_loader (overfit_check.py --sample-mode live parity); "
                    f"val_loader shrunk to a fixed subset as before")
        val_loader = _shrink_dataset_for_smoke(val_loader, max_samples=10)

    model = RIUnet(in_channels=cfg.model_config["IN_CHANNELS"]).to(device)

    weights_dtype_str = cfg_train.get("WEIGHTS_DTYPE", None)
    if weights_dtype_str:
        wd = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(weights_dtype_str.lower())
        if wd and wd != torch.float32:
            model = model.to(dtype=wd)

    pw_cap = float(cfg_train.get("POS_WEIGHT_CAP", 1000.0))
    pos_weight = estimate_pos_weight(train_loader, max_batches=int(cfg_train.get("POS_WEIGHT_BATCHES", 20)), cap=pw_cap)
    
    # --- Loss Setup & Logging ---
    loss_fn = _build_loss(cfg_train, pos_weight)
    loss_type = str(cfg_train.get("LOSS_TYPE", "DICECE")).upper()
    logger.info(f"Loss Function: {loss_type} | pos_weight: {pos_weight:.2f} (cap: {pw_cap}) | Class: {loss_fn.__class__.__name__}")

    # --- Optimizer & LR Logging ---
    base_lr = float(cfg_train["LR"])
    weight_decay = float(cfg_train["WEIGHT_DECAY"])
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=False)

    save_config_snapshot(run_paths["run"], extra={"pos_weight": pos_weight})

    # --- Scheduler Setup & Logging ---
    epochs = int(cfg_train["EPOCHS"])
    warmup_epochs = int(cfg_train.get("WARMUP_EPOCHS", 0))
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = LambdaLR(optimizer, lr_lambda=lambda e: min((e + 1) / warmup_epochs, 1.0))
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    start_epoch, best_val_dice, bad_epochs = 0, -1.0, 0
    if args.resume:
        last = run_paths["ckpts"] / "best_dice.pth"
        logger.info(f"Attempting to resume from {last}")
        if last.exists():
            # 1. Load file once to extract state and metrics efficiently
            checkpoint_data = torch.load(last, map_location="cpu", weights_only=False)
            
            # 2. Restore weights, optimizer, and scaler
            model.load_state_dict(checkpoint_data["model"])
            if "optimizer" in checkpoint_data and optimizer is not None:
                optimizer.load_state_dict(checkpoint_data["optimizer"])
            if "scaler" in checkpoint_data and scaler is not None:
                scaler.load_state_dict(checkpoint_data["scaler"])
                
            # 3. Extract epoch and metric state
            start_epoch = int(checkpoint_data.get("epoch", -1)) + 1
            metrics = checkpoint_data.get("metrics", {})
            best_val_dice = float(metrics.get("best_val_dice", metrics.get("val_dice", -1.0)))
            
            # 4. Fast-forward LR scheduler to sync with start_epoch
            for _ in range(start_epoch):
                scheduler.step()

            logger.info(f"Resumed from {last} at epoch {start_epoch} | Restored best_val_dice: {best_val_dice:.4f} | Current LR: {optimizer.param_groups[0]['lr']:.6e}")
        else:
            logger.warning(f"Resume path {last} does not exist. Starting training from scratch.")


    logger.info(f"Training Config | Total Epochs: {epochs} | Warmup Epochs: {warmup_epochs} | Base LR: {base_lr} | Weight Decay: {weight_decay} | AMP: {cfg_train.get('AMP', True)} | GradScaler enabled: {scaler.is_enabled()}")

    
    if args.smoke_test and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    cpu_t0, t0 = proc.cpu_times(), time.perf_counter()

    history = []
    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, epoch, cfg_train,
                                       run_paths, logger, n_samples=smoke_n_samples)
        scheduler.step()
        val_stats = validate(model, val_loader, loss_fn, device, cfg_train)
        
        logger.info(f"  epoch {epoch:03d} | lr {current_lr:.6e} | val loss {val_stats['val_loss']:.4f} | val dice {val_stats['val_dice']:.4f}")
        history.append({"epoch": epoch, "lr": current_lr, **train_stats, **val_stats})

        is_best = val_stats["val_dice"] > best_val_dice
        if is_best:
            best_val_dice, bad_epochs = val_stats["val_dice"], 0
        else:
            bad_epochs += 1

        save_checkpoint(model, optimizer, scaler, epoch,
                         metrics={**train_stats, **val_stats, "best_val_dice": best_val_dice, "lr": current_lr},
                         run_paths=run_paths, is_best=is_best)

        patience = int(cfg_train.get("EARLY_STOP_PATIENCE", 0))
        if patience > 0 and bad_epochs >= patience:
            logger.info("early stop condition met.")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1, cpu_t1 = time.perf_counter(), proc.cpu_times()

    if args.smoke_test:
        total_time = t1 - t0
        cpu_time = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
        peak_gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
        logger.info("\n" + "=" * 60)
        logger.info(" SMOKE TEST RESOURCE METRICS")
        logger.info("=" * 60)
        logger.info(f"  Wall Time    : {total_time:.2f} s")
        logger.info(f"  CPU Time     : {cpu_time:.2f} s")
        logger.info(f"  Peak GPU Mem : {peak_gpu_mem:.1f} MB")
        logger.info("=" * 60)

    with open(run_paths["run"] / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=float)
    logger.info(f"=== complete. best val_dice = {best_val_dice:.4f} ===")


if __name__ == "__main__":
    main()