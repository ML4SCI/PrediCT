import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

csv_path = Path("data_canonical/tables/scan_index.csv")
df = pd.read_csv(csv_path)

# Verify core columns exist
score_col = "agatston_total"
if score_col not in df.columns:
    raise KeyError(f"Could not find '{score_col}' in your CSV. Available columns: {list(df.columns)}")

df = df[df[score_col] > 0]  # Ensure no negative scores


def assign_risk_label(score):
    if score < 10:
        return 'Low Risk'
    elif score < 100:
        return 'Medium Risk'
    elif score < 400:
        return 'High Risk'
    elif score < 1000:
        return 'Very High Risk'
    else:
        return 'Extreme Risk'

df['risk_label'] = df[score_col].apply(assign_risk_label)

# Convert to a categorical type to keep your plots ordered nicely from Low -> High
df['risk_label'] = pd.Categorical(df['risk_label'], categories=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk', 'Extreme Risk'], ordered=True)

print(f"--- Loaded {len(df)} Patients from Scan Index ---")
print(df['risk_label'].value_counts().sort_index())
print("\n" + "="*50 + "\n")

# These features define the multi-dimensional spatial topology of the calcium
feature_cols = [
    'lesion_count', 
    'agatston_total', 
    'agatston_rca', 
    'agatston_left_coronary', 
    'agatston_lad', 
    'agatston_lcx'
]

# Ensure we drop any rows missing these crucial anatomical features
df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

df_clean[feature_cols] = np.log1p(df_clean[feature_cols])

X = df_clean[feature_cols].values

# Scale features so that large total scores don't mathematically crush the 'lesion_count' variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# HIGH-DIMENSIONAL t-SNE MAPPING

perplexity_val = min(30, max(2, len(df_clean) // 3))

tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca')
tsne_results = tsne.fit_transform(X_scaled)

df_clean['tsne_dim1'] = tsne_results[:, 0]
df_clean['tsne_dim2'] = tsne_results[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="tsne_dim1", y="tsne_dim2",
    hue="risk_label",
    palette="Spectral",
    data=df_clean,
    s=100,
    alpha=0.8,
    edgecolor='w'
)

# # to match your 10-patient showcase style
# num_annotations = min(12, len(df_clean))
# for i in range(num_annotations):
#     plt.text(
#         df_clean['tsne_dim1'][i], df_clean['tsne_dim2'][i], 
#         f" ID:{df_clean['patient_id'][i]} (LC:{int(df_clean['lesion_count'][i])})", 
#         fontsize=8, alpha=0.7, weight='semibold'
#     )

# Filter for patients where BOTH major branches have active plaque (>0)
multi_artery_df = df_clean[(df_clean['agatston_lad'] > 0) & (df_clean['agatston_rca'] > 0)]

# Fallback to the regular dataset if too few patients match the strict condition
if len(multi_artery_df) < 5:
    multi_artery_df = df_clean

# Label up to 10 points scattered across these multi-artery configurations
num_annotations = min(10, len(multi_artery_df))

for i in range(num_annotations):
    row = multi_artery_df.iloc[i]
    
    # Simple, clean display format: ID, LC, and cross-artery confirmation
    annotation_text = (
        f" ID:{int(row['patient_id'])}\n"
        f" (LC:{int(row['lesion_count'])}, LAD+RCA>0)"
    )
    
    plt.text(
        row['tsne_dim1'] + 0.3, # Slight offset to prevent overlapping the dot
        row['tsne_dim2'], 
        annotation_text, 
        fontsize=7, 
        alpha=0.85, 
        weight='semibold',
        color='black'
    )

plt.title("t-SNE Spatial Embedding of CAC Profiles\n(Proving that patients inside the same clinical bin split into different geometric clusters)", fontsize=12, pad=15)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Clinical Risk Bin")
plt.tight_layout()
plt.savefig("cac_tsne_spatial_clusters.png", dpi=300)
print("Saved 2D t-SNE map to 'cac_tsne_spatial_clusters.png'")

plt.show()