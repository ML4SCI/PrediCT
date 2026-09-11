import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------

csv_path = Path("data_canonical/tables/scan_index.csv")
df = pd.read_csv(csv_path)

score_col = "agatston_total"

if score_col not in df.columns:
    raise KeyError(
        f"Could not find '{score_col}' in CSV."
    )

# Keep CAC-positive only
df = df[df[score_col] > 0].reset_index(drop=True)

# ----------------------------------------------------------
# Risk labels
# ----------------------------------------------------------

def assign_risk_label(score):
    if score < 10:
        return "Low Risk"
    elif score < 100:
        return "Medium Risk"
    elif score < 400:
        return "High Risk"
    elif score < 1000:
        return "Very High Risk"
    else:
        return "Extreme Risk"


df["risk_label"] = df[score_col].apply(assign_risk_label)

df["risk_label"] = pd.Categorical(
    df["risk_label"],
    categories=[
        "Low Risk",
        "Medium Risk",
        "High Risk",
        "Very High Risk",
        "Extreme Risk",
    ],
    ordered=True,
)

print(f"\nLoaded {len(df)} patients\n")
print(df["risk_label"].value_counts().sort_index())

# ----------------------------------------------------------
# Features
# ----------------------------------------------------------

feature_cols = [
    "lesion_count",
    "agatston_total",
    "agatston_rca",
    "agatston_left_coronary",
    "agatston_lad",
    "agatston_lcx",
]

df_clean = (
    df
    .dropna(subset=feature_cols)
    .reset_index(drop=True)
)

# ----------------------------------------------------------
# RAW FEATURES
# ----------------------------------------------------------

X_raw = df_clean[feature_cols].copy()

X_raw_scaled = StandardScaler().fit_transform(X_raw)

# ----------------------------------------------------------
# LOG FEATURES
# ----------------------------------------------------------

X_log = df_clean[feature_cols].copy()

X_log[feature_cols] = np.log1p(
    X_log[feature_cols]
)

X_log_scaled = StandardScaler().fit_transform(
    X_log
)

# ----------------------------------------------------------
# t-SNE
# ----------------------------------------------------------

perplexity = min(
    30,
    max(2, len(df_clean) // 3)
)

tsne_raw = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42,
    init="pca",
)

tsne_log = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42,
    init="pca",
)

raw_embedding = tsne_raw.fit_transform(
    X_raw_scaled
)

log_embedding = tsne_log.fit_transform(
    X_log_scaled
)

df_clean["raw_x"] = raw_embedding[:, 0]
df_clean["raw_y"] = raw_embedding[:, 1]

df_clean["log_x"] = log_embedding[:, 0]
df_clean["log_y"] = log_embedding[:, 1]

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 8)
)

palette = "Spectral"

sns.scatterplot(
    data=df_clean,
    x="raw_x",
    y="raw_y",
    hue="risk_label",
    palette=palette,
    s=80,
    edgecolor="white",
    alpha=0.85,
    ax=axes[0],
)

axes[0].set_title(
    "Raw Features"
)

sns.scatterplot(
    data=df_clean,
    x="log_x",
    y="log_y",
    hue="risk_label",
    palette=palette,
    s=80,
    edgecolor="white",
    alpha=0.85,
    ax=axes[1],
)

axes[1].set_title(
    "Log1p Transformed Features"
)

# ----------------------------------------------------------
# Annotate same patients
# ----------------------------------------------------------

multi = df_clean[
    (df_clean["agatston_lad"] > 0)
    &
    (df_clean["agatston_rca"] > 0)
]

if len(multi) < 10:
    multi = df_clean

annotate_df = multi.iloc[:10]

for _, row in annotate_df.iterrows():

    label = (
        f"ID:{int(row['patient_id'])}\n"
        f"LC:{int(row['lesion_count'])}"
    )

    axes[0].text(
        row["raw_x"] + 0.3,
        row["raw_y"],
        label,
        fontsize=7,
    )

    axes[1].text(
        row["log_x"] + 0.3,
        row["log_y"],
        label,
        fontsize=7,
    )

# ----------------------------------------------------------
# Legend
# ----------------------------------------------------------

axes[0].legend_.remove()

handles, labels = axes[1].get_legend_handles_labels()

axes[1].legend(
    handles,
    labels,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    title="Clinical Risk Bin",
)

plt.suptitle(
    "t-SNE Spatial Embedding of CAC Profiles\n"
    "Raw Features vs Log1p Features",
    fontsize=15,
)

plt.tight_layout()

plt.savefig(
    "tsne_raw_vs_log.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()