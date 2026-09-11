"""
model_sanity_check.py — Isolate MODEL x LOSS combinations from the DATA
PIPELINE, using synthetic data with sparsity matching your real dataset.

Grid this script lets you sweep:
    --model {unet, riunet}
    --loss  {dice_ce, broken_dice_ce, weighted_dice_ce, tversky_ce}

Why synthetic data:
    overfit_check.py tests model + loss + real dataloader together. If it
    fails, you can't tell whether the bug is in model/loss wiring or
    somewhere in dataset.py / labels / augmentation. This script removes the
    real data entirely: one SYNTHETIC sample with sparsity matching your
    real data (~0.001-0.004% foreground, same as your logged fg_voxels
    counts) so difficulty is comparable, and iterates fast (no dataloader
    startup, no I/O).

Interpretation:
    - PASS here (mean_prob@fg -> ~1, mean_prob@bg -> ~0) but overfit_check.py
      still fails on real data -> bug is in the data pipeline (dataset.py,
      label loading/alignment, augmentation), NOT model or loss wiring.
    - FAIL here too -> the bug/limitation is in the model/loss/optimizer
      combo itself. Compare across the --loss grid to find which one(s)
      actually converge before touching dataset.py at all.

`mean_prob@fg` and `mean_prob@bg` should SEPARATE, not both rise:
    fg -> 1.0 (confident foreground where label says foreground)
    bg -> 0.0 (confident background everywhere else)
If both rise together the model is predicting foreground everywhere; if
both fall together (your current symptom) it's predicting background
everywhere. Either way, watch the two numbers pull apart, not just move.

The `broken_dice_ce` option deliberately reproduces a specific suspected
real bug: constructing DiceCELoss(softmax=True, ...) WITHOUT sigmoid=True.
For a single-channel model, MONAI's own warning tells you `softmax=True`
is ignored on single-channel predictions -- meaning NO activation gets
applied at all, and Dice ends up computed directly on raw unbounded logits
as if they were already probabilities. That produces a stable but
meaningless ~0.5-ish loss, which matches the plateau you've been seeing.
Compare `broken_dice_ce` vs `dice_ce` (which explicitly sets sigmoid=True)
side by side to confirm whether this is your real bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceLoss, TverskyLoss
from monai.networks.nets import UNet

# So `from model import RIUnet` works when this script sits next to your
# project files (same layout as overfit_check.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_synthetic_sample(
    in_channels: int,
    roi=(128, 128, 32),
    n_fg_voxels: int = 100,
    device="cuda",
    seed: int = 0,
):
    """One synthetic (image, label) pair matching your real data's shape and
    foreground sparsity (~0.001-0.004% in your logs)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, in_channels, *roi, generator=g).to(device)

    y = torch.zeros(1, 1, *roi, device=device)
    flat = y.view(-1)
    idx = torch.randperm(flat.numel(), generator=g)[:n_fg_voxels]
    flat[idx] = 1.0
    return x, y


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_unet(in_channels: int, device="cuda") -> nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    ).to(device)


def build_riunet(in_channels: int, device="cuda") -> nn.Module:
    """Your own model. Same construction call as the commented-out line in
    overfit_check.py: RIUnet(in_channels=...). If your RIUnet takes
    additional required args (out_channels, features, etc.), add them here
    to match how it's actually constructed in train.py."""
    try:
        from model import RIUnet
    except ImportError as e:
        raise ImportError(
            "Could not import RIUnet from model.py — run this script from "
            "the same directory as model.py, or adjust sys.path."
        ) from e
    return RIUnet(in_channels=in_channels).to(device)


MODEL_BUILDERS = {
    "unet": build_unet,
    "riunet": build_riunet,
}


# --------------------------------------------------------------------------
# Losses — all take (logits, target) and return a scalar
# --------------------------------------------------------------------------

def make_loss(loss_kind: str, pos_weight: float, device: str):
    if loss_kind == "dice_ce":
        # Correctly wired: sigmoid=True is explicit, so single-channel
        # output actually gets an activation applied before Dice/CE.
        fn = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
        return fn

    if loss_kind == "broken_dice_ce":
        # Deliberately reproduces the suspected real bug: softmax=True on a
        # single-channel model. MONAI's own warning says softmax=True is
        # IGNORED for single-channel predictions -> no activation applied
        # at all -> Dice computed on raw unbounded logits. Kept here as a
        # control to compare against `dice_ce` above.
        fn = DiceCELoss(softmax=True, lambda_dice=0.5, lambda_ce=0.5)
        return fn

    if loss_kind == "weighted_dice_ce":
        pw = torch.tensor(pos_weight, device=device)
        dice = DiceLoss(sigmoid=True)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)

        def fn(pred, target):
            return 0.5 * dice(pred, target.float()) + 0.5 * bce(pred, target.float())

        return fn

    if loss_kind == "tversky_ce":
        pw = torch.tensor(pos_weight, device=device)
        # alpha < beta weights false negatives more heavily than false
        # positives -- pushes the model to prefer catching sparse
        # foreground over staying safely all-background.
        tversky = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)

        def fn(pred, target):
            return 0.5 * tversky(pred, target.float()) + 0.5 * bce(pred, target.float())

        return fn

    raise ValueError(f"unknown loss_kind: {loss_kind}")


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def run_one(
    model_kind: str,
    loss_kind: str,
    pos_weight: float,
    iters: int,
    lr: float,
    in_channels: int,
    n_fg_voxels: int,
    log_every: int,
    device: str,
):
    torch.manual_seed(42)
    x, y = make_synthetic_sample(in_channels, n_fg_voxels=n_fg_voxels, device=device)
    n_total = y.numel()

    print("\n" + "#" * 70)
    print(f"# model={model_kind}  loss={loss_kind}  pos_weight={pos_weight}  lr={lr}")
    print("#" * 70)
    print(
        f"synthetic sample: {tuple(y.shape)} | fg_voxels={n_fg_voxels}/{n_total} "
        f"({100.0 * n_fg_voxels / n_total:.4f}%)"
    )

    model = MODEL_BUILDERS[model_kind](in_channels, device=device)

    with torch.no_grad():
        raw_out = model(x)
        raw_out = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    print(
        f"[logits check] raw output range: min={raw_out.min().item():.4f} "
        f"max={raw_out.max().item():.4f} mean={raw_out.mean().item():.4f} "
        f"(should NOT be confined to [0,1] -- confirms raw logits, no "
        f"internal activation)"
    )

    loss_fn = make_loss(loss_kind, pos_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()

    final_fg = final_bg = float("nan")
    for it in range(iters):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        out = out[0] if isinstance(out, (list, tuple)) else out
        loss = loss_fn(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % log_every == 0 or it == iters - 1:
            with torch.no_grad():
                probs = torch.sigmoid(out)
                fg_mean = probs[y > 0].mean().item()
                bg_mean = probs[y <= 0].mean().item()
            final_fg, final_bg = fg_mean, bg_mean
            print(
                f"  iter {it:04d} | loss {loss.item():.6f} | "
                f"mean_prob@fg={fg_mean:.4f} mean_prob@bg={bg_mean:.4f}"
            )

    verdict = "PASS" if (final_fg > 0.9 and final_bg < 0.1) else "FAIL"
    print(f"-> {verdict}: mean_prob@fg={final_fg:.4f}  mean_prob@bg={final_bg:.4f}")
    return {
        "model": model_kind,
        "loss": loss_kind,
        "final_fg": final_fg,
        "final_bg": final_bg,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="all", choices=["unet", "riunet", "all"],
        help="Which model(s) to test. 'all' runs unet then riunet.",
    )
    ap.add_argument(
        "--loss", default="all",
        choices=["dice_ce", "broken_dice_ce", "weighted_dice_ce", "tversky_ce", "all"],
        help="Which loss(es) to test. 'all' runs the full grid.",
    )
    ap.add_argument("--pos-weight", type=float, default=1000.0)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--in-channels", type=int, default=5)
    ap.add_argument("--n-fg-voxels", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    models = ["unet", "riunet"] if args.model == "all" else [args.model]
    losses = (
        ["dice_ce", "broken_dice_ce", "weighted_dice_ce", "tversky_ce"]
        if args.loss == "all"
        else [args.loss]
    )

    results = []
    for m in models:
        for l in losses:
            try:
                res = run_one(
                    model_kind=m,
                    loss_kind=l,
                    pos_weight=args.pos_weight,
                    iters=args.iters,
                    lr=args.lr,
                    in_channels=args.in_channels,
                    n_fg_voxels=args.n_fg_voxels,
                    log_every=args.log_every,
                    device=args.device,
                )
            except Exception as e:
                print(f"\n[ERROR] model={m} loss={l} raised: {e}")
                res = {"model": m, "loss": l, "final_fg": float("nan"), "final_bg": float("nan"), "verdict": "ERROR"}
            results.append(res)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'model':10} {'loss':18} {'fg':>8} {'bg':>8}   verdict")
    for r in results:
        print(
            f"{r['model']:10} {r['loss']:18} {r['final_fg']:8.4f} "
            f"{r['final_bg']:8.4f}   {r['verdict']}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()