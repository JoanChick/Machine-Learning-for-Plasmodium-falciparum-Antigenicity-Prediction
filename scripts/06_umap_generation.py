#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
import logging
import sys
import warnings
warnings.filterwarnings('ignore')

import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SPLITS_DIR   = Path("splits/selected")
PLOTS_DIR    = Path("plots/umap")
RESULTS_DIR  = Path("results")
RANDOM_STATE = 42

UMAP_PARAMS = dict(
    n_neighbors  = 15,
    min_dist     = 0.1,
    n_components = 2,
    random_state = RANDOM_STATE,
)

COLORS = {0: "#E74C3C", 1: "#2ECC71"}
LABELS = {0: "Non-Antigen", 1: "Antigen"}

FEATURE_GROUPS = {
    "Length":                 ["ln_"],
    "Amino Acid Composition": ["aac_"],
    "AAIndex1":               ["aaind_"],
    "Dipeptide":              ["di_"],
    "Tripeptide":             ["tri_"],
    "Entropy":                ["ent_"],
    "Atom Composition":       ["atc_"],
    "Binary Encoding":        ["bin_"],
    "Position-Rich":          ["posrich_"],
    "CKSAAP (k=1)":           ["ck1_"],
    "CKSAAP (k=2)":           ["ck2_"],
    "CTD":                    ["ctd_"],
    "CTDC":                   ["ctdc_"],
    "CTDT":                   ["ctdt_"],
    "CTDD":                   ["ctdd_"],
    "Moreau-Broto":           ["mb_"],
    "Moran":                  ["moran_"],
    "Geary":                  ["geary_"],
    "PAAC":                   ["paac_"],
    "APAAC":                  ["apaac_"],
    "SOCN":                   ["socn_"],
    "QSO":                    ["qso_"],
    "ALL FEATURES":           None,    # None = use all 1000 selected features
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"umap_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Load and merge all splits
# ──────────────────────────────────────────────
def load_all_data():
    log.info(f"Loading selected splits from '{SPLITS_DIR}' ...")
    dfs = []
    for split in ["train", "validation", "test"]:
        dfs.append(pd.read_csv(SPLITS_DIR / f"{split}_selected.csv"))
    df = pd.concat(dfs, ignore_index=True)
    y  = df["label"].values
    X  = df.drop(columns=["label"])
    log.info(f"  Total samples  : {len(y)}")
    log.info(f"  Total features : {X.shape[1]}")
    log.info(f"  Positives      : {y.sum()}  |  Negatives: {(y==0).sum()}")
    return X, y


# ──────────────────────────────────────────────
# 2. Select features belonging to a group
# ──────────────────────────────────────────────
def select_group(X: pd.DataFrame, prefixes) -> np.ndarray:
    if prefixes is None:
        return X.values
    cols = [c for c in X.columns if any(c.startswith(p) for p in prefixes)]
    if len(cols) == 0:
        return None
    return X[cols].values


# ──────────────────────────────────────────────
# 3. PCA pre-reduction if group is too wide
#    Speeds up UMAP and avoids degenerate embeddings
# ──────────────────────────────────────────────
def safe_reduce(X_group: np.ndarray, name: str) -> np.ndarray:
    n_samples, n_feats = X_group.shape
    if n_feats > 500 or n_feats >= n_samples:
        n_comp = min(50, n_samples - 1, n_feats)
        log.info(f"  PCA pre-reduction: {n_feats} -> {n_comp} dims")
        X_group = PCA(n_components=n_comp,
                      random_state=RANDOM_STATE).fit_transform(X_group)
    return X_group


# ──────────────────────────────────────────────
# 4. Scale and run UMAP
# ──────────────────────────────────────────────
def run_umap(X_group: np.ndarray) -> np.ndarray:
    X_scaled  = StandardScaler().fit_transform(X_group)
    embedding = umap.UMAP(**UMAP_PARAMS).fit_transform(X_scaled)
    return embedding


# ──────────────────────────────────────────────
# 5. Per-group scatter plot
# ──────────────────────────────────────────────
def plot_umap(embedding: np.ndarray, y: np.ndarray,
              group_name: str, silhouette: float,
              n_features: int, out_path: Path):

    fig, ax = plt.subplots(figsize=(7, 6))

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=COLORS[cls], label=LABELS[cls],
            alpha=0.55, s=18, edgecolors='none', rasterized=True
        )

    # Cluster centre stars
    for cls in [0, 1]:
        mask   = y == cls
        centre = embedding[mask].mean(axis=0)
        ax.scatter(*centre, c=COLORS[cls], s=150,
                   marker='*', edgecolors='black',
                   linewidths=0.8, zorder=5)

    sil_color = "#27AE60" if silhouette > 0.3 else \
                "#E67E22" if silhouette > 0.1 else "#C0392B"

    ax.set_title(f'UMAP - {group_name}\n'
                 f'({n_features} features selected | '
                 f'Silhouette = {silhouette:.3f})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP-1', fontsize=10)
    ax.set_ylabel('UMAP-2', fontsize=10)
    ax.legend(handles=[
        mpatches.Patch(color=COLORS[c], label=LABELS[c]) for c in [0, 1]
    ], fontsize=10)
    ax.grid(alpha=0.2)
    ax.text(0.98, 0.02, f'Silhouette: {silhouette:.3f}',
            transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=sil_color,
                      alpha=0.25, edgecolor=sil_color))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────
# 6. Summary grid — all groups ranked by silhouette
# ──────────────────────────────────────────────
def plot_summary_grid(results: list, out_path: Path):
    results_sorted = sorted(results, key=lambda r: r["silhouette"],
                            reverse=True)
    n     = len(results_sorted)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(ncols * 5, nrows * 4.5))
    fig.suptitle(
        "UMAP - All Feature Groups (1000 SelectKBest features)\n"
        "Ranked by Silhouette Score",
        fontsize=13, fontweight='bold', y=1.01
    )

    for idx, res in enumerate(results_sorted):
        ax  = fig.add_subplot(nrows, ncols, idx + 1)
        emb = res["embedding"]
        y   = res["y"]
        sil = res["silhouette"]

        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=COLORS[cls], alpha=0.45, s=8,
                       edgecolors='none', rasterized=True)

        sil_color = "#27AE60" if sil > 0.3 else \
                    "#E67E22" if sil > 0.1 else "#C0392B"
        ax.set_title(f'{res["group"]}\n'
                     f'Sil={sil:.3f} | {res["n_features"]}f',
                     fontsize=8, fontweight='bold', color=sil_color)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    handles = [mpatches.Patch(color=COLORS[c], label=LABELS[c])
               for c in [0, 1]]
    fig.legend(handles=handles, loc='lower center',
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Summary grid saved -> {out_path}")


# ──────────────────────────────────────────────
# 7. Silhouette ranking bar chart
# ──────────────────────────────────────────────
def plot_silhouette_ranking(results: list, out_path: Path):
    results_sorted = sorted(results, key=lambda r: r["silhouette"],
                            reverse=True)
    groups = [r["group"]      for r in results_sorted]
    scores = [r["silhouette"] for r in results_sorted]
    colors = ["#27AE60" if s > 0.3 else
              "#E67E22" if s > 0.1 else
              "#C0392B" for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(5, len(groups) * 0.45)))
    bars = ax.barh(range(len(groups)), scores,
                   color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=9)
    ax.set_xlabel("Silhouette Score", fontsize=11)
    ax.set_title(
        "Feature Group Separability — UMAP Silhouette Score\n"
        "(1000 SelectKBest-selected features)\n"
        "Green > 0.3 = good | Orange > 0.1 = moderate | Red = poor",
        fontsize=11, fontweight='bold'
    )
    ax.axvline(0.3, color='green',  linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(0.1, color='orange', linestyle='--', linewidth=1, alpha=0.6)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, scores):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Silhouette ranking saved -> {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_all_data()

    results      = []
    summary_rows = []

    for group_name, prefixes in FEATURE_GROUPS.items():
        log.info(f"\n{'-'*55}")
        log.info(f"  Processing: {group_name}")

        X_group = select_group(X, prefixes)
        if X_group is None or X_group.shape[1] == 0:
            log.warning(f"  No features found for '{group_name}' "
                        f"in selected set -- skipping")
            continue

        n_features = X_group.shape[1]
        log.info(f"  Features in selected set: {n_features}")

        X_group   = safe_reduce(X_group, group_name)
        embedding = run_umap(X_group)
        sil       = silhouette_score(embedding, y)
        log.info(f"  Silhouette score: {sil:.4f}")

        safe_name = (group_name
                     .replace(" ", "_")
                     .replace("(", "").replace(")", "")
                     .replace("=", ""))
        out_path  = PLOTS_DIR / f"umap_{safe_name}.png"
        plot_umap(embedding, y, group_name, sil, n_features, out_path)
        log.info(f"  Saved -> {out_path}")

        results.append({
            "group":      group_name,
            "n_features": n_features,
            "silhouette": sil,
            "embedding":  embedding,
            "y":          y,
        })
        summary_rows.append({
            "feature_group": group_name,
            "n_features":    n_features,
            "silhouette":    round(sil, 4),
        })

    # Summary plots
    log.info("\nGenerating summary plots ...")
    plot_summary_grid(
        results, PLOTS_DIR / "umap_all_groups_grid.png")
    plot_silhouette_ranking(
        results, PLOTS_DIR / "umap_silhouette_ranking.png")

    # Save scores table
    summary_df = (pd.DataFrame(summary_rows)
                  .sort_values("silhouette", ascending=False)
                  .reset_index(drop=True))
    summary_df["rank"] = range(1, len(summary_df) + 1)
    out_csv = RESULTS_DIR / "umap_silhouette_scores.csv"
    summary_df.to_csv(out_csv, index=False)
    log.info(f"\n  Silhouette scores saved -> {out_csv}")

    log.info("\n" + "="*55)
    log.info("UMAP SILHOUETTE RANKING")
    log.info("="*55)
    log.info(f"  {'Rank':<6} {'Group':<28} {'Silhouette':<12} {'N Features'}")
    log.info(f"  {'-'*55}")
    for _, row in summary_df.iterrows():
        sil_label = "good"     if row["silhouette"] > 0.3 else \
                    "moderate" if row["silhouette"] > 0.1 else "poor"
        log.info(f"  {int(row['rank']):<6} {row['feature_group']:<28} "
                 f"{row['silhouette']:<12.4f} {int(row['n_features'])} "
                 f"[{sil_label}]")
    log.info("="*55)
    log.info("UMAP analysis complete.")


if __name__ == "__main__":
    main()
