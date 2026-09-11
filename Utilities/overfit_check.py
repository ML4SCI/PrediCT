"""
overfit_check.py — Sanity-check that the model/loss/data pipeline can overfit
a tiny fixed subset before trusting "convergence" judgments on the full run.

Why this matters:
    A flat train loss + near-zero val dice can mean either (a) genuine
    optimization/architecture trouble, or (b) a metric/channel-mapping bug
    that makes the *reported* dice meaningless even when the model is
    learning fine. Overfitting 1-2 samples is a cheap way to tell them apart:
    if the model CANNOT drive loss to ~0 and dice to ~1 on a handful of
    samples with no augmentation, something structural is broken (data,
    labels, loss config, or the model itself) — it is not a "needs more
    epochs / more data" problem.

CLI:
    python overfit_check.py EXPERIMENT_NAME
    python overfit_check.py EXPERIMENT_NAME --n-samples 2 --epochs 300
    python overfit_check.py EXPERIMENT_NAME --lr 1e-3
    python overfit_check.py EXPERIMENT_NAME --weighted-ce --pos-weight-cap 100

Notes:
    - Reuses your real dataset/model/loss construction (build_dataloaders,
      RIUnet, _build_loss) so this is testing your *actual* pipeline, not a
      toy stand-in.
    - Uses the SAME fixed subset every epoch, no shuffling, so any failure
      to memorize can't be blamed on data churn.
    - Runs plain full-batch/no-sliding-window forward passes on the small
      crop images directly (not sliding_window_inference) since these are
      small fixed samples we want to iterate over fast.
    - Model is single-channel (out_channels=1) -> sigmoid, NOT softmax/argmax.
      compute_dice and the debug logit stats below both assume sigmoid.
    - At the end, prints a sigmoid-probability-at-known-foreground-voxel
      check to help sanity-check learning directly from a trained (in this
      case, over-trained) checkpoint.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceLoss, TverskyLoss
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

import config as cfg
from dataset import build_dataloaders
from model import RIUnet

# Reuse the loss builder + DS loss helper from train.py so the loss config
# matches your real training exactly.
from train import _build_loss, ds_loss, set_seed, get_logger, SPLITS_JSON

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"


def build_fixed_subset(loader, n_samples: int) -> List[dict]:
    """Pull the first n_samples batches out of `loader` ONCE and cache them
    in memory (as CPU tensors) so every epoch trains on the identical data."""
    fixed = []
    it = iter(loader)
    for _ in range(n_samples):
        try:
            batch = next(it)
        except StopIteration:
            break
        fixed.append(batch)
    if not fixed:
        raise RuntimeError("Loader produced zero batches — check dataset/splits.")
    return fixed


def compute_pos_weight(fixed_batches: List[dict], cap: float = 100.0) -> float:
    """Background:foreground voxel ratio across the fixed subset, capped so
    it doesn't destabilize training when foreground is this sparse (ratios
    of 1:1000+ are common here — using the raw ratio as pos_weight blows up
    the loss scale and tends to diverge rather than help)."""
    total_fg = total_vox = 0
    for batch in fixed_batches:
        y = batch["label"]
        total_fg += int((y > 0).sum().item())
        total_vox += int(y.numel())
    total_bg = total_vox - total_fg
    if total_fg == 0:
        return cap
    return float(min(total_bg / total_fg, cap))


class WeightedDiceCE(nn.Module):
    """Dice + pos_weight-BCE for single-channel (sigmoid) binary segmentation.

    Plain unweighted BCE gives ~0 effective gradient signal from the
    foreground class when it's this sparse (a handful of voxels vs. millions
    of background voxels) — the network can minimize loss almost entirely by
    getting background right. pos_weight scales up the loss contribution of
    positive (foreground) voxels so missing them actually costs something.
    """

    def __init__(self, pos_weight: float, lambda_dice: float = 0.5, lambda_ce: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.register_buffer("pos_weight_t", torch.tensor(pos_weight))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_t)
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, pred, target):
        target = target.float()
        return self.lambda_dice * self.dice(pred, target) + self.lambda_ce * self.bce(pred, target)


class TverskyWeightedCE(nn.Module):
    """Tversky + pos_weight-BCE, single-channel (sigmoid).

    In the RIUnet synthetic sanity check, this converged faster than plain
    weighted Dice+BCE (loss ~0.03 by iter 50 vs ~0.14) on data this sparse.
    alpha<beta weights false negatives more heavily than false positives,
    which pushes harder toward catching rare foreground voxels than Dice
    alone does.
    """

    def __init__(
        self,
        pos_weight: float,
        alpha: float = 0.3,
        beta: float = 0.7,
        lambda_tversky: float = 0.5,
        lambda_ce: float = 0.5,
    ):
        super().__init__()
        self.tversky = TverskyLoss(sigmoid=True, alpha=alpha, beta=beta)
        self.register_buffer("pos_weight_t", torch.tensor(pos_weight))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_t)
        self.lambda_tversky = lambda_tversky
        self.lambda_ce = lambda_ce

    def forward(self, pred, target):
        target = target.float()
        return (
            self.lambda_tversky * self.tversky(pred, target)
            + self.lambda_ce * self.bce(pred, target)
        )


@torch.no_grad()
def compute_dice(out: torch.Tensor, y: torch.Tensor, device) -> float:
    """Dice for a single-channel (sigmoid) model.

    IMPORTANT: with out_channels=1, `out.argmax(dim=1)` is ALWAYS index 0
    (argmax over a size-1 dimension is a no-op that always returns 0), which
    silently forces the predicted mask to be all-False forever regardless of
    what the model learns. That was the previous bug here — dice was
    mathematically guaranteed to read 0.0 no matter how well training went.
    Use sigmoid + threshold instead.
    """
    pred_mask = torch.sigmoid(out) > 0.5
    target_mask = y > 0

    intersection = (pred_mask & target_mask).sum()
    total = pred_mask.sum() + target_mask.sum()

    dice = torch.where(
        total > 0, (2.0 * intersection) / total, torch.tensor(1.0, device=device)
    )
    return float(dice.item())


@torch.no_grad()
def debug_logit_stats(out: torch.Tensor, y: torch.Tensor, logger) -> None:
    """Sigmoid probability stats, split by whether the voxel is foreground
    or background in the label. This is the signal to watch even before
    dice moves off zero: mean_prob@fg should rise toward 1 and
    mean_prob@bg should stay near 0 as training progresses."""
    probs = torch.sigmoid(out)
    fg_mask, bg_mask = y > 0, y <= 0
    fg_mean = probs[fg_mask].mean().item() if fg_mask.any() else float("nan")
    bg_mean = probs[bg_mask].mean().item() if bg_mask.any() else float("nan")
    logger.info(
        f"    [probs] min={probs.min().item():.4f} mean={probs.mean().item():.4f} "
        f"max={probs.max().item():.4f} | mean_prob@fg={fg_mean:.4f} mean_prob@bg={bg_mean:.4f}"
    )


@torch.no_grad()
def channel_order_check(out: torch.Tensor, y: torch.Tensor) -> None:
    """Inspect the sigmoid probability at a real foreground voxel (per label
    map) after overfitting. For a single-channel model there's no
    "channel order" to confuse — this instead confirms the model actually
    assigns a high probability to a voxel it was supposed to memorize."""
    fg_idx = (y[0, 0] > 0).nonzero()
    if fg_idx.numel() == 0:
        print("  [CHANNEL CHECK] No foreground voxels found in y[0,0] — skipping.")
        return

    i, j, k = fg_idx[0].tolist()
    logit = out[0, 0, i, j, k].item()
    prob = torch.sigmoid(out[0, 0, i, j, k]).item()

    print("\n" + "=" * 60)
    print(" SINGLE-CHANNEL SIGMOID CHECK (at one known foreground voxel)")
    print("=" * 60)
    print(f"  voxel index (i,j,k) = ({i}, {j}, {k})")
    print(f"  logit  = {logit:.4f}")
    print(f"  sigmoid(logit) = {prob:.4f}")
    print(
        "  -> after overfitting, this should be close to 1.0 "
        "(model confidently predicts foreground here)."
    )
    print("=" * 60 + "\n")


def build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    """Same warmup->cosine pattern as train.py's main(). If warmup_epochs<=0
    or >= epochs, falls back to plain cosine annealing over the full run."""
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = LambdaLR(optimizer, lr_lambda=lambda e: min((e + 1) / warmup_epochs, 1.0))
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

def overfit_run(
    experiment_name: str,
    n_samples: int,
    epochs: int,
    lr: float,
    log_every: int,
    warmup_epochs: int,
    use_scheduler: bool,
    loss_kind: str,
    pos_weight_cap: float,
    sample_mode: str = "cached",
) -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = RUNS_DIR / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(run_dir / "overfit_check.log")
    logger.info(f"=== overfit_check | experiment: {experiment_name} | device: {device} | sample_mode: {sample_mode} ===")

    logger.info("building dataloaders…")
    train_loader, _, _ = build_dataloaders(str(SPLITS_JSON))

    cfg.model_config["IN_CHANNELS"] = int(
        next(iter(train_loader))["image"].shape[1]
    )
    logger.info(f"auto-detected IN_CHANNELS = {cfg.model_config['IN_CHANNELS']}")

    # Setup batch sampling strategy
    if sample_mode == "cached":
        fixed_batches = build_fixed_subset(train_loader, n_samples)
        logger.info(f"cached {len(fixed_batches)} fixed sample(s) in memory for overfitting")
        
        # Report foreground stats for cached batches up front
        for idx, batch in enumerate(fixed_batches):
            y = batch["label"]
            n_fg = int((y > 0).sum().item())
            n_total = int(y.numel())
            logger.info(
                f"  sample {idx}: label shape={tuple(y.shape)} "
                f"fg_voxels={n_fg} / {n_total} ({100.0 * n_fg / n_total:.4f}%)"
            )
        pos_weight_calc_batches = fixed_batches
    else:
        logger.info(f"using live train_loader pipeline: sampling {n_samples} batch(es) per epoch on-the-fly")
        # Pull a quick initial set of batches just to calculate pos_weight if needed
        pos_weight_calc_batches = [next(iter(train_loader)) for _ in range(n_samples)]

    model = RIUnet(in_channels=cfg.model_config["IN_CHANNELS"]).to(device)

    if loss_kind in ("weighted_dice_ce", "tversky_ce"):
        pos_weight = compute_pos_weight(pos_weight_calc_batches, cap=pos_weight_cap)
        logger.info(
            f"loss_kind={loss_kind}: pos_weight={pos_weight:.2f} (capped at {pos_weight_cap})"
        )
        if loss_kind == "weighted_dice_ce":
            loss_fn = WeightedDiceCE(pos_weight=pos_weight).to(device)
        else:
            loss_fn = TverskyWeightedCE(pos_weight=pos_weight).to(device)
    else:
        logger.info("loss_kind=default: using _build_loss(cfg.train_config) as-is")
        loss_fn = _build_loss(dict(cfg.train_config))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = GradScaler("cuda", enabled=False)

    scheduler = build_scheduler(optimizer, epochs, warmup_epochs) if use_scheduler else None

    ds_level_weights = cfg.train_config.get("DS_LEVEL_WEIGHTS", [1.0])

    model.train()
    t0 = time.time()
    history = []

    for epoch in range(epochs):
        epoch_losses = []
        epoch_dices = []
        last_out = last_y = None

        # Determine batch stream for this epoch
        if sample_mode == "cached":
            batch_iterator = fixed_batches
        else:
            # Yield n_samples batches directly from the live train_loader
            def _live_generator():
                for step, batch in enumerate(train_loader):
                    if step >= n_samples:
                        break
                    yield batch
            batch_iterator = _live_generator()

        for batch in batch_iterator:
            x = batch["image"].to(device, non_blocking=True).to(torch.float32)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            out = model(x)
            loss = ds_loss(out, y, loss_fn, ds_level_weights)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(float(loss.item()))

            main_out = out[0] if isinstance(out, (list, tuple)) else out
            epoch_dices.append(compute_dice(main_out.detach(), y, device))

            last_out, last_y = main_out.detach(), y

        mean_loss = float(np.mean(epoch_losses))
        mean_dice = float(np.mean(epoch_dices))
        history.append({"epoch": epoch, "loss": mean_loss, "dice": mean_dice})

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % log_every == 0 or epoch == epochs - 1:
            logger.info(
                f"  epoch {epoch:04d} | loss {mean_loss:.6f} | dice {mean_dice:.6f} | lr {current_lr:.2e}"
            )
            debug_logit_stats(last_out, last_y, logger)

        if scheduler is not None:
            scheduler.step()

    dt = time.time() - t0
    logger.info(f"done in {dt:.1f}s")

    final_loss = history[-1]["loss"]
    final_dice = history[-1]["dice"]
    logger.info(f"\nFINAL: loss={final_loss:.6f}  dice={final_dice:.6f}")

    # Sigmoid check on the last available batch
    with torch.no_grad():
        channel_order_check(last_out, last_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_name", help="Run name; creates runs/<name>/overfit_check.log")
    ap.add_argument("--n-samples", type=int, default=1, help="How many fixed samples to overfit on")
    ap.add_argument("--epochs", type=int, default=300, help="Number of epochs to overfit for")
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for the overfit run (often higher than normal training LR)")
    ap.add_argument("--log-every", type=int, default=10, help="Log every N epochs")
    ap.add_argument(
        "--scheduler", action="store_true",
        help="Enable warmup->cosine LR schedule (same pattern as train.py) instead of a fixed LR.",
    )
    ap.add_argument(
        "--warmup-epochs", type=int, default=0,
        help="Warmup epochs before cosine annealing kicks in (only used with --scheduler).",
    )
    ap.add_argument(
        "--loss-kind", default="default",
        choices=["default", "weighted_dice_ce", "tversky_ce"],
        help=(
            "default: _build_loss(cfg.train_config) as-is. "
            "weighted_dice_ce / tversky_ce: pos_weight-based losses confirmed "
            "to converge on RIUnet in model_sanity_check.py's synthetic test."
        ),
    )
    ap.add_argument(
        "--pos-weight-cap", type=float, default=1000.0,
        help="Cap on the computed pos_weight (bg/fg voxel ratio). Only used with weighted_dice_ce/tversky_ce.",
    )
    ap.add_argument(
        "--sample-mode",
        choices=["cached", "live"],
        default="cached",
        help=(
            "cached: pre-extract N static batches in memory once (pure memorization check). "
            "live: pull N batches per epoch directly through train_loader (tests live MONAI transform/crop pipeline)."
        ),
    )
    args = ap.parse_args()

    overfit_run(
        experiment_name=args.experiment_name,
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        log_every=args.log_every,
        warmup_epochs=args.warmup_epochs,
        use_scheduler=args.scheduler,
        loss_kind=args.loss_kind,
        pos_weight_cap=args.pos_weight_cap,
        sample_mode=args.sample_mode
    )


if __name__ == "__main__":
    main()