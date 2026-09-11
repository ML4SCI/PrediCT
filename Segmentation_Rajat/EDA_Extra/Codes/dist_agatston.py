import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path("data_canonical/tables/scan_index.csv"))

# Assuming the column name is 'agatston_score' or similar (adjust if different)
# Let's find it dynamically if it's named slightly differently
score_col = "agatston_total"

# --- 1. Print Frequencies (Zero vs. Positive Cases) ---
total_scans = len(df)
zero_count = (df[score_col] == 0).sum()
positive_count = (df[score_col] > 0).sum()

print("="*40)
print(f"FREQUENCY BREAKDOWN FOR: {score_col}")
print("="*40)
print(f"Zero CAC (Score == 0): {zero_count:4d} ({zero_count/total_scans*100:5.2f}%)")
print(f"With CAC (Score >  0): {positive_count:4d} ({positive_count/total_scans*100:5.2f}%)")
print(f"Total Scans:          {total_scans:4d}")
print("-"*40)

# Optional: Print standard clinical severity tiers for the positive cases
if positive_count > 0:
    pos_data = df[df[score_col] > 0][score_col]
    mild = ((pos_data > 0) & (pos_data <= 10)).sum()
    moderate = ((pos_data > 10) & (pos_data <= 100)).sum()
    severe = ((pos_data > 100) & (pos_data <= 400)).sum()
    critical = ((pos_data > 400) & (pos_data <= 1000)).sum()
    extreme = (pos_data > 1000).sum()

    print("Positive Case Breakdown (Clinical Tiers):")
    print(f"  Minimal (0 - 10]:    {mild:4d} ({mild/positive_count*100:5.2f}%)")
    print(f"  Moderate (10 - 100]: {moderate:4d} ({moderate/positive_count*100:5.2f}%)")
    print(f"  Severe (100 - 400]:  {severe:4d} ({severe/positive_count*100:5.2f}%)")
    print(f"  Critical (400 - 1000]: {critical:4d} ({critical/positive_count*100:5.2f}%)")
    print(f"  Extreme (> 1000):    {extreme:4d} ({extreme/positive_count*100:5.2f}%)")
    print("="*40)

# --- 2. Plot Distribution ---
plt.figure(figsize=(10, 6))

plt.hist(df[df[score_col] > 0][score_col], bins=50, color='crimson', edgecolor='black', alpha=0.7, log=True)

plt.title(f'Distribution of Agatston Scores[LOG] (Score>0 Filtered)', fontsize=14, fontweight='bold')
plt.xlabel('Agatston Score', fontsize=12)
plt.ylabel('Frequency[LOG]', fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

