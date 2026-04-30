#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (QuadraticDiscriminantAnalysis,
                                            LinearDiscriminantAnalysis)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SPLITS_DIR   = Path("splits/selected")
RESULTS_DIR  = Path("results/feature_importance")
PLOTS_DIR    = Path("plots/feature_importance")
RANDOM_STATE = 0
PERM_REPEATS = 30      # permutation importance repeats - higher = more stable

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
}

MODELS = {
    "SVM":      SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
    "RF":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost":  XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss',
                               verbosity=0),
    "LR":       LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "NB":       GaussianNB(),
    "DT":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE),
    "QDA":      QuadraticDiscriminantAnalysis(reg_param=0.5),
    "LDA":      LinearDiscriminantAnalysis(),
    "ET":       ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GB":       GradientBoostingClassifier(n_estimators=100,
                                            random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
    "kNN":      KNeighborsClassifier(n_neighbors=5),
}

# Model type routing
TREE_BASED  = {"RF", "ET", "XGBoost", "LightGBM", "GB", "AdaBoost", "DT"}
LINEAR      = {"LR", "SVM", "LDA"}
PERM_BASED  = {"NB", "QDA", "kNN"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"feature_importance_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
def load_data():
    log.info(f"Loading data from '{SPLITS_DIR}' ...")

    train_df = pd.read_csv(SPLITS_DIR / "train_selected.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "validation_selected.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test_selected.csv")

    # Combine train+val for fitting (same as CV script)
    cv_df    = pd.concat([train_df, val_df], ignore_index=True)
    X_cv     = cv_df.drop(columns=["label"]).values
    y_cv     = cv_df["label"].values
    X_test   = test_df.drop(columns=["label"]).values
    y_test   = test_df["label"].values
    features = cv_df.drop(columns=["label"]).columns.tolist()

    log.info(f"  CV pool  : {X_cv.shape[0]} samples x {X_cv.shape[1]} features")
    log.info(f"  Test set : {X_test.shape[0]} samples")
    log.info(f"  Features : {len(features)}")
    return X_cv, y_cv, X_test, y_test, features


# ──────────────────────────────────────────────
# 2. Assign each feature to its biological group
# ──────────────────────────────────────────────
def assign_groups(features: list) -> pd.Series:
    group_map = {}
    for feat in features:
        assigned = "Other"
        for group, prefixes in FEATURE_GROUPS.items():
            if any(feat.startswith(p) for p in prefixes):
                assigned = group
                break
        group_map[feat] = assigned
    return pd.Series(group_map, name="group")


# ──────────────────────────────────────────────
# 3. Extract importance scores by model type
# ──────────────────────────────────────────────
def get_importances(name: str, clf, X_cv, y_cv,
                    X_test, y_test, features: list) -> np.ndarray:
    """Returns a 1-D array of importance scores, one per feature."""

    if name in TREE_BASED:
        log.info(f"  Method: native feature_importances_")
        return clf.feature_importances_

    elif name in LINEAR:
        log.info(f"  Method: |coef_| (linear weights)")
        coef = clf.coef_
        if coef.ndim > 1:
            # Multi-class or single row - take mean of absolute values
            return np.abs(coef).mean(axis=0)
        return np.abs(coef).flatten()

    else:
        log.info(f"  Method: permutation importance "
                 f"(n_repeats={PERM_REPEATS}) on test set")
        result = permutation_importance(
            clf, X_test, y_test,
            n_repeats    = PERM_REPEATS,
            random_state = RANDOM_STATE,
            scoring      = "roc_auc",
            n_jobs       = -1,
        )
        # Clip negatives to 0 - negative means feature adds noise
        return np.clip(result.importances_mean, 0, None)


# ──────────────────────────────────────────────
# 4. Aggregate importances to group level
# ──────────────────────────────────────────────
def aggregate_groups(importances: np.ndarray,
                     features: list,
                     group_map: pd.Series) -> pd.DataFrame:
    feat_df = pd.DataFrame({
        "feature":    features,
        "importance": importances,
        "group":      group_map.values,
    })

    group_df = (feat_df
                .groupby("group")["importance"]
                .agg(
                    total_importance = "sum",
                    mean_importance  = "mean",
                    max_importance   = "max",
                    n_features       = "count",
                )
                .reset_index()
                .sort_values("total_importance", ascending=False)
                .reset_index(drop=True))

    # Normalise total importance to sum to 1 (percentage contribution)
    group_df["contribution_pct"] = (
        group_df["total_importance"] /
        group_df["total_importance"].sum() * 100
    ).round(2)

    group_df["rank"] = range(1, len(group_df) + 1)
    return group_df, feat_df


# ──────────────────────────────────────────────
# 5. Plots
# ──────────────────────────────────────────────
def plot_group_barplot(group_df: pd.DataFrame, name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn(
        np.linspace(0.8, 0.2, len(group_df)))  # green=top, red=bottom

    bars = ax.barh(
        group_df["group"][::-1],
        group_df["contribution_pct"][::-1],
        color=colors[::-1],
        edgecolor='black', linewidth=0.5
    )

    for bar, val in zip(bars, group_df["contribution_pct"][::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=8)

    ax.set_xlabel('Contribution to Model (%)', fontsize=11)
    ax.set_title(f'Feature Group Importance Ranking - {name}',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_group_barplot.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_all_models_heatmap(all_group_importance: dict,
                             out_dir: Path):
    """
    Heatmap: rows = feature groups, cols = models
    Values = contribution_pct (normalised so each model sums to 100%)
    """
    # Build matrix
    all_groups = sorted(FEATURE_GROUPS.keys())
    model_names = list(all_group_importance.keys())

    matrix = pd.DataFrame(index=all_groups, columns=model_names, dtype=float)
    for model_name, group_df in all_group_importance.items():
        for _, row in group_df.iterrows():
            if row["group"] in matrix.index:
                matrix.loc[row["group"], model_name] = row["contribution_pct"]
    matrix = matrix.fillna(0)

    # Sort rows by mean contribution across models
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        matrix,
        annot=True, fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor='grey',
        ax=ax,
        cbar_kws={"label": "Contribution (%)"},
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Group Contribution (%) per Model\n"
                 "(sorted by mean contribution across all models)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Model",         fontsize=11)
    ax.set_ylabel("Feature Group", fontsize=11)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "all_models_group_heatmap.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Heatmap saved → {out_dir / 'all_models_group_heatmap.png'}")


def plot_consensus_ranking(all_group_importance: dict, out_dir: Path):
    """
    Bar chart of mean contribution % across all models -
    shows which feature groups are consistently important.
    """
    all_groups  = sorted(FEATURE_GROUPS.keys())
    model_names = list(all_group_importance.keys())

    records = []
    for model_name, group_df in all_group_importance.items():
        for _, row in group_df.iterrows():
            records.append({
                "group": row["group"],
                "contribution_pct": row["contribution_pct"]
            })

    consensus = (pd.DataFrame(records)
                 .groupby("group")["contribution_pct"]
                 .agg(mean="mean", std="std")
                 .reset_index()
                 .sort_values("mean", ascending=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(consensus)))

    ax.barh(consensus["group"][::-1],
            consensus["mean"][::-1],
            xerr=consensus["std"][::-1],
            color=colors[::-1],
            edgecolor='black', linewidth=0.5,
            error_kw=dict(ecolor='black', capsize=3, linewidth=1))

    ax.set_xlabel('Mean Contribution Across All Models (%)', fontsize=11)
    ax.set_title('Consensus Feature Group Ranking\n'
                 '(mean ± std contribution across all 14 models)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "consensus_group_ranking.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Consensus ranking saved → "
             f"{out_dir / 'consensus_group_ranking.png'}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_cv, y_cv, X_test, y_test, features = load_data()
    group_map = assign_groups(features)

    log.info(f"\n  Feature group coverage:")
    for group, prefixes in FEATURE_GROUPS.items():
        count = sum(1 for f in features
                    if any(f.startswith(p) for p in prefixes))
        log.info(f"  {group:<28}: {count} features")

    all_group_importance = {}   # model_name -> group_df
    all_feat_rows        = []
    all_group_rows       = []

    for name, clf in MODELS.items():
        model_plot_dir = PLOTS_DIR / name
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*55}")
        log.info(f"  Model: {name}")
        log.info(f"{'='*55}")

        try:
            # Train on full CV pool
            log.info(f"  Training on CV pool ...")
            clf.fit(X_cv, y_cv)

            # Get importances
            importances = get_importances(
                name, clf, X_cv, y_cv, X_test, y_test, features)

            # Aggregate to groups
            group_df, feat_df = aggregate_groups(
                importances, features, group_map)

            # Add model column
            feat_df["model"]  = name
            group_df["model"] = name
            all_feat_rows.append(feat_df)
            all_group_rows.append(group_df)
            all_group_importance[name] = group_df

            # Log ranking
            log.info(f"\n  Feature group ranking for {name}:")
            log.info(f"  {'Rank':<6} {'Group':<28} {'Contribution%':<15} "
                     f"{'N Features'}")
            log.info(f"  {'-'*60}")
            for _, row in group_df.iterrows():
                log.info(f"  {int(row['rank']):<6} {row['group']:<28} "
                         f"{row['contribution_pct']:<15.2f} "
                         f"{int(row['n_features'])}")

            # Save per-model CSVs
            feat_df.to_csv(
                RESULTS_DIR / f"{name}_feature_importance.csv", index=False)
            group_df.to_csv(
                RESULTS_DIR / f"{name}_group_importance.csv",   index=False)

            # Per-model bar plot
            plot_group_barplot(group_df, name, model_plot_dir)
            log.info(f"  Saved → '{model_plot_dir}/'")

        except Exception as e:
            log.error(f"  {name} FAILED: {e}")
            continue

    # ── Combined outputs ──
    log.info("\nGenerating combined plots ...")

    pd.concat(all_feat_rows,  ignore_index=True).to_csv(
        RESULTS_DIR / "all_models_feature_importance.csv", index=False)
    pd.concat(all_group_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "all_models_group_importance.csv",   index=False)

    plot_all_models_heatmap(all_group_importance, PLOTS_DIR)
    plot_consensus_ranking(all_group_importance,  PLOTS_DIR)

    # ── Consensus ranking table ──
    records = []
    for model_name, group_df in all_group_importance.items():
        for _, row in group_df.iterrows():
            records.append({
                "group":            row["group"],
                "contribution_pct": row["contribution_pct"],
            })
    consensus = (pd.DataFrame(records)
                 .groupby("group")["contribution_pct"]
                 .agg(mean="mean", std="std")
                 .reset_index()
                 .sort_values("mean", ascending=False)
                 .reset_index(drop=True))
    consensus["rank"] = range(1, len(consensus) + 1)
    consensus.to_csv(RESULTS_DIR / "consensus_group_ranking.csv", index=False)

    log.info("\n" + "="*55)
    log.info("CONSENSUS FEATURE GROUP RANKING (across all models)")
    log.info("="*55)
    log.info(f"  {'Rank':<6} {'Group':<28} {'Mean %':<12} {'Std %'}")
    log.info(f"  {'-'*55}")
    for _, row in consensus.iterrows():
        log.info(f"  {int(row['rank']):<6} {row['group']:<28} "
                 f"{row['mean']:<12.2f} {row['std']:.2f}")
    log.info("="*55)
    log.info("Feature group importance analysis complete.")


if __name__ == "__main__":
    main()
