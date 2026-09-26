"""
correlate_population_atlases.py

Step 3 of the Population Cohort Approach.
This script overlays the Healthy ESS Atlas and the Diseased Calcium Atlas
in the standardized Reference space to prove the statistical correlation.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.spatial as spatial
from pathlib import Path

OUTPUT_DIR = Path("/Users/karan/Desktop/PrediCT/output_population")

def main():
    print("==========================================================")
    print("  STEP 3: CORRELATE POPULATION ATLASES")
    print("==========================================================")
    
    ess_path = OUTPUT_DIR / "ess_atlas.csv"
    calc_path = OUTPUT_DIR / "calcium_atlas.csv"
    
    if not ess_path.exists() or not calc_path.exists():
        print("Error: Missing population atlases. Run Step 1 and 2 first.")
        return
        
    print("Loading Population Atlases...")
    ess_df = pd.read_csv(ess_path)
    calc_df = pd.read_csv(calc_path)
    
    print(f"Loaded {len(ess_df)} ESS points (Healthy Cohort)")
    print(f"Loaded {len(calc_df)} Calcium points (Diseased Cohort)")
    
    if len(ess_df) == 0 or len(calc_df) == 0:
        print("Not enough points to correlate.")
        return
        
    # Build KD-Tree for spatial mapping
    print("Building spatial KD-Tree...")
    ess_coords = ess_df[["x", "y", "z"]].values
    calc_coords = calc_df[["x", "y", "z"]].values
    
    tree = spatial.cKDTree(ess_coords)
    
    # For every calcium point, find the nearest healthy ESS value
    print("Mapping diseased calcium to healthy ESS distributions...")
    dists, indices = tree.query(calc_coords, k=1)
    
    # Filter out calcium points that mapped too far from any artery
    # (Because Phase 1 macroscopic registration isn't sub-millimeter perfect,
    # we allow a spatial tolerance, e.g. 15mm, to capture the epicardial fat radius)
    valid_mask = dists < 15.0
    valid_calcium_ess = ess_df.iloc[indices[valid_mask]]["ess"].values
    
    print(f"Successfully mapped {np.sum(valid_mask)} calcium deposits to local hemodynamics.")
    
    # Baseline ESS distribution (all points)
    baseline_ess = ess_df["ess"].values
    
    # --- STATISTICS ---
    mean_baseline = np.mean(baseline_ess)
    mean_calcium = np.mean(valid_calcium_ess)
    
    from scipy.stats import mannwhitneyu
    stat, p_val = mannwhitneyu(valid_calcium_ess, baseline_ess, alternative='less')
    
    print("\n--- POPULATION CORRELATION RESULTS ---")
    print(f"Mean ESS (Healthy Artery Baseline)    : {mean_baseline:.3f} Pa")
    print(f"Mean ESS (Sites of Future Calcium)    : {mean_calcium:.3f} Pa")
    print(f"Mann-Whitney U Test p-value           : {p_val:.2e}")
    
    if mean_calcium < mean_baseline and p_val < 0.05:
        print(">>> CONCLUSION: VALIDATION PASSED. Calcium forms in statistically lower ESS zones. <<<")
    else:
        print(">>> CONCLUSION: NO SIGNIFICANT CORRELATION FOUND. <<<")
        
    # --- VISUALIZATION ---
    print("\nGenerating Statistical Visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Density plot of ESS
    import seaborn as sns
    sns.kdeplot(baseline_ess, ax=axes[0], fill=True, color="blue", label="All Coronary Tissue", alpha=0.3)
    sns.kdeplot(valid_calcium_ess, ax=axes[0], fill=True, color="red", label="Calcified Tissue", alpha=0.5)
    axes[0].set_xlabel("Physiological Endothelial Shear Stress (Pa)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("ESS Distribution: Healthy vs. Diseased Sites")
    axes[0].legend()
    axes[0].axvline(mean_baseline, color="blue", linestyle="--")
    axes[0].axvline(mean_calcium, color="red", linestyle="--")
    
    # 2. Probability of Calcium by ESS Band
    bands = np.linspace(0, max(np.max(baseline_ess), 2.0), 10)
    calc_hist, _ = np.histogram(valid_calcium_ess, bins=bands)
    base_hist, _ = np.histogram(baseline_ess, bins=bands)
    
    # Probability = P(Calcium | ESS band)
    prob = calc_hist / (base_hist + 1e-9)
    # Normalize for plotting
    prob = prob / np.max(prob)
    
    axes[1].bar(bands[:-1], prob, width=(bands[1]-bands[0])*0.8, align='edge', color="purple", alpha=0.7)
    axes[1].set_xlabel("Endothelial Shear Stress (Pa)")
    axes[1].set_ylabel("Relative Risk of Calcium Formation")
    axes[1].set_title("Atherosclerotic Risk vs. Shear Stress")
    
    fig.tight_layout()
    out_fig = OUTPUT_DIR / "population_correlation.png"
    fig.savefig(out_fig, dpi=200)
    print(f"Saved visualization to {out_fig}")

if __name__ == "__main__":
    main()
