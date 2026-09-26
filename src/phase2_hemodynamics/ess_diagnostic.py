"""
ESS Diagnostic & Validation Script
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

def generate_diagnostics(csv_path: str, out_dir: str) -> None:
    CSV_PATH = Path(csv_path)
    OUT_DIR  = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    ess   = df["ess_magnitude"].values
    x, y, z = df["x"].values, df["y"].values, df["z"].values
    wss_x, wss_y, wss_z = df["wss_x"].values, df["wss_y"].values, df["wss_z"].values
    wss_mag = np.sqrt(wss_x**2 + wss_y**2 + wss_z**2)
    N = len(ess)
    print(f"Loaded {N} wall points\n")

    physio_low, physio_high = 1.0, 7.0
    q1, q3 = np.percentile(ess, [25, 75])
    iqr = q3 - q1
    outlier_threshold = q3 + 3 * iqr
    mask_out = ess > outlier_threshold
    mask_nor = ~mask_out
    n_outliers = mask_out.sum()

    print("=" * 55)
    print("  ESS DISTRIBUTION STATISTICS")
    print("=" * 55)
    for lbl, v in zip(
        ["Mean","Median","Std","Min","p5","p25","p75","p90","p95","p99","Max"],
        [ess.mean(), np.median(ess), ess.std(), ess.min(),
         *np.percentile(ess,[5,25,75,90,95,99]), ess.max()]
    ):
        print(f"  {lbl:<7}: {v:.6f} Pa")
    print()

    frac_low    = (ess < physio_low).mean() * 100
    frac_normal = ((ess >= physio_low) & (ess <= physio_high)).mean() * 100
    frac_high   = (ess > physio_high).mean() * 100
    print("  PHYSIOLOGICAL BAND BREAKDOWN")
    print(f"  ESS < 1.0 Pa (atherogenic):  {frac_low:5.1f}%")
    print(f"  1.0–7.0 Pa  (normal):        {frac_normal:5.1f}%")
    print(f"  ESS > 7.0 Pa (erosion risk): {frac_high:5.1f}%")
    print(f"\n  Outlier threshold (Q3+3IQR): {outlier_threshold:.3f} Pa")
    print(f"  N outliers: {n_outliers} / {N} ({n_outliers/N*100:.1f}%)\n")

    DARK, PANEL, TEXT = "#1a1a2e", "#16213e", "#e0e0e0"
    BLUE, ACCENT, GOLD = "#0f3460", "#e94560", "#f7b731"

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)

    def sax(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor("#334466")
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

    # Panel 1 – full histogram log
    ax1 = fig.add_subplot(gs[0, 0]); sax(ax1)
    ax1.hist(ess, bins=80, color="#4cc9f0", edgecolor="none", alpha=0.85, log=True)
    ax1.axvline(physio_low,  color=ACCENT, lw=1.5, ls="--", label="1 Pa")
    ax1.axvline(physio_high, color=GOLD,   lw=1.5, ls="--", label="7 Pa")
    ax1.axvline(ess.mean(),  color="#a8ff78", lw=1.5, label=f"Mean={ess.mean():.3f}")
    ax1.set_xlabel("ESS (Pa)"); ax1.set_ylabel("Count (log)")
    ax1.set_title("ESS Distribution (all points)")
    ax1.legend(fontsize=7, facecolor=BLUE, labelcolor=TEXT)

    # Panel 2 – bulk clipped at 10 Pa
    ax2 = fig.add_subplot(gs[0, 1]); sax(ax2)
    bulk = ess[ess <= 10.0]
    ax2.hist(bulk, bins=60, color="#4cc9f0", edgecolor="none", alpha=0.85)
    ax2.axvline(physio_low,  color=ACCENT, lw=1.5, ls="--", label="1 Pa")
    ax2.axvline(physio_high, color=GOLD,   lw=1.5, ls="--", label="7 Pa")
    ax2.axvline(np.median(ess), color="#a8ff78", lw=1.5, label=f"Median={np.median(ess):.3f}")
    ax2.set_xlabel("ESS (Pa)"); ax2.set_ylabel("Count")
    ax2.set_title(f"Bulk ESS <=10 Pa  ({len(bulk)/N*100:.0f}% of points)")
    ax2.legend(fontsize=7, facecolor=BLUE, labelcolor=TEXT)

    # Panel 3 – spatial map with outliers highlighted
    ax3 = fig.add_subplot(gs[0, 2]); sax(ax3)
    sc = ax3.scatter(x[mask_nor], z[mask_nor], c=ess[mask_nor],
                      cmap="plasma", s=7, alpha=0.65, vmin=0, vmax=physio_high)
    if n_outliers > 0:
        ax3.scatter(x[mask_out], z[mask_out], c=ACCENT, s=30, marker="x",
                     alpha=0.9, label=f"Outliers >{outlier_threshold:.1f} Pa (N={n_outliers})")
        ax3.legend(fontsize=7, facecolor=BLUE, labelcolor=TEXT)
    cb = plt.colorbar(sc, ax=ax3)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=8)
    cb.set_label("ESS (Pa)", color=TEXT, fontsize=9)
    ax3.set_xlabel("x (non-dim)"); ax3.set_ylabel("z (non-dim)")
    ax3.set_title("Spatial ESS Map (x-z, outliers in red)")

    # Panel 4 – WSS vs ESS (should be 1:1)
    ax4 = fig.add_subplot(gs[1, 0]); sax(ax4)
    ax4.scatter(wss_mag, ess, s=4, alpha=0.4, color="#4cc9f0")
    lim = min(max(wss_mag.max(), ess.max()), 15)
    ax4.plot([0, lim], [0, lim], color=ACCENT, lw=1.2, label="1:1 line")
    ax4.set_xlim(0, lim); ax4.set_ylim(0, lim)
    ax4.set_xlabel("||WSS vector|| (Pa)"); ax4.set_ylabel("ESS magnitude (Pa)")
    ax4.set_title("WSS vector vs ESS (1:1 sanity check)")
    ax4.legend(fontsize=7, facecolor=BLUE, labelcolor=TEXT)

    # Panel 5 – ESS vs axial position
    ax5 = fig.add_subplot(gs[1, 1]); sax(ax5)
    sc5 = ax5.scatter(x, ess, c=z, cmap="cool", s=5, alpha=0.55)
    ax5.axhline(physio_low,  color=ACCENT, lw=1.2, ls="--", label="1 Pa")
    ax5.axhline(physio_high, color=GOLD,   lw=1.2, ls="--", label="7 Pa")
    ax5.set_xlabel("x (axial, non-dim)"); ax5.set_ylabel("ESS (Pa)")
    ax5.set_title("ESS vs Axial Position")
    cb5 = plt.colorbar(sc5, ax=ax5)
    cb5.set_label("z (non-dim)", color=TEXT, fontsize=8)
    cb5.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    ax5.legend(fontsize=7, facecolor=BLUE, labelcolor=TEXT)

    # Panel 6 – CDF
    ax6 = fig.add_subplot(gs[1, 2]); sax(ax6)
    sorted_ess = np.sort(ess); cdf = np.arange(1, N+1) / N
    ax6.plot(sorted_ess, cdf, color="#4cc9f0", lw=2)
    ax6.axvline(physio_low,  color=ACCENT, lw=1.5, ls="--", label="1 Pa")
    ax6.axvline(physio_high, color=GOLD,   lw=1.5, ls="--", label="7 Pa")
    ax6.set_xlabel("ESS (Pa)"); ax6.set_ylabel("Cumulative fraction")
    ax6.set_title("ESS Cumulative Distribution")
    ax6.set_xlim(0, min(15, sorted_ess.max()))
    ax6.legend(fontsize=7, facecolor=BLUE, labelcolor=TEXT)

    fig.suptitle(
        f"ESS Diagnostic Report  —  N={N}  |  Mean={ess.mean():.3f} Pa  |  Median={np.median(ess):.3f} Pa",
        color=TEXT, fontsize=13, y=0.99
    )
    out_fig = OUT_DIR / "ess_diagnostic.png"
    fig.savefig(out_fig, dpi=180, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Figure saved -> {out_fig}")

    print("\n  TOP-10 HIGHEST ESS POINTS (outlier inspection)")
    for i, idx in enumerate(np.argsort(ess)[-10:][::-1]):
        print(f"  #{i+1:2d}  ESS={ess[idx]:8.3f} Pa  x={x[idx]:.3f}  y={y[idx]:.3f}  z={z[idx]:.3f}")

    near_inlet  = (x[mask_out] < x.min() + 0.5).sum() if n_outliers else 0
    near_outlet = (x[mask_out] > x.max() - 0.5).sum() if n_outliers else 0
    print(f"\n  Outliers near inlet        : {near_inlet}")
    print(f"  Outliers near outlet       : {near_outlet}")
    print(f"  Outliers in interior       : {max(0, n_outliers - near_inlet - near_outlet)}")

    mu, U_ref, L_ref = 0.0035, 0.25, 0.003
    wss_theory = mu * 2 * (U_ref/2) / (L_ref/2)
    print(f"\n  Poiseuille WSS (theory)  : {wss_theory:.4f} Pa")
    print(f"  PINN mean ESS            : {ess.mean():.4f} Pa")
    print(f"  Ratio PINN/theory        : {ess.mean()/wss_theory:.3f}")

    with open(OUT_DIR / "validation_report.txt", "w") as f:
        f.write("--- ESS VALIDATION REPORT ---\n")
        f.write(f"Mean ESS: {ess.mean():.4f} Pa\n")
        f.write(f"Ratio PINN/theory: {ess.mean()/wss_theory:.3f}\n")

    print("Diagnostics complete.")

if __name__ == "__main__":
    generate_diagnostics(
        "/Users/karan/Desktop/PrediCT/output_v2/exports/ess_predictions.csv",
        "/Users/karan/Desktop/PrediCT/output_v2/exports/diagnostics"
    )
