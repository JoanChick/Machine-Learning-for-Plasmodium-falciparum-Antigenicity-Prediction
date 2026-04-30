#!/usr/bin/env python3

import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_DIR   = Path("splits")
RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("plots")

VARIANCE_THRESH    = 0.00
CORRELATION_THRESH = 0.95

# K values to search — log-spaced to cover small and large ranges efficiently
K_CANDIDATES = [50, 100, 150, 200, 300, 400, 500, 600, 750, 1000,
                1250, 1500, 2000, 2500, 3000]

CV_FOLDS     = 10       # stratified k-fold on training set
SCORING      = "roc_auc"   # change to "f1" or "matthews_corrcoef" if preferred
RANDOM_STATE = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"optimal_k_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Load & pre-filter training data
#    (variance + correlation — same as 01_feature_selection.py
#     so K search operates on the same reduced pool)
# ──────────────────────────────────────────────
def load_train() -> tuple[np.ndarray, np.ndarray, list]:
    df   = pd.read_csv(INPUT_DIR / "train.csv")
    y    = df["label"].values
    X    = df.drop(columns=["label"]).values
    cols = df.drop(columns=["label"]).columns.tolist()
    log.info(f"  Train loaded: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, cols


def prefilter(X_train, cols):
    """Apply variance + correlation filter — same settings as feature selection."""
    log.info("\n[Pre-filter] Variance Threshold ...")
    var_sel = VarianceThreshold(threshold=VARIANCE_THRESH)
    var_sel.fit(X_train)
    mask  = var_sel.get_support()
    X     = X_train[:, mask]
    cols  = [c for c, m in zip(cols, mask) if m]
    log.info(f"  After variance: {len(cols)} features")

    log.info("[Pre-filter] Correlation Filter ...")
    df    = pd.DataFrame(X, columns=cols)
    corr  = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        to_drop.update(upper.index[upper[col] > CORRELATION_THRESH].tolist())
    keep  = [c not in to_drop for c in cols]
    cols  = [c for c, k in zip(cols, keep) if k]
    X     = df.loc[:, keep].values
    log.info(f"  After correlation: {len(cols)} features")

    return X, cols


# ──────────────────────────────────────────────
# 2. Cross-validate a Pipeline(SelectKBest → LR)
#    for each K candidate
# ──────────────────────────────────────────────
def search_k(X_train, y_train, max_features: int) -> pd.DataFrame:
    log.info(f"\n[K Search] Scoring metric: {SCORING} | CV folds: {CV_FOLDS}")

    # Only test K values that don't exceed available features
    candidates = [k for k in K_CANDIDATES if k <= max_features]
    log.info(f"  K candidates: {candidates}")

    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    rows    = []

    for k in candidates:
        pipeline = Pipeline([
            ("select",  SelectKBest(score_func=f_classif, k=k)),
            ("scale",   StandardScaler()),          # LR needs scaling
            ("clf",     LogisticRegression(
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                            class_weight="balanced" # handles imbalance
                        )),
        ])

        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=True,
            n_jobs=-1
        )

        mean_val   = cv_results["test_score"].mean()
        std_val    = cv_results["test_score"].std()
        mean_train = cv_results["train_score"].mean()

        rows.append({
            "k":          k,
            "val_mean":   round(mean_val,   4),
            "val_std":    round(std_val,    4),
            "train_mean": round(mean_train, 4),
            "overfit_gap": round(mean_train - mean_val, 4),
        })

        log.info(f"  K={k:<5} | val {SCORING}={mean_val:.4f} ± {std_val:.4f} "
                 f"| train={mean_train:.4f} | gap={mean_train-mean_val:.4f}")

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 3. Pick optimal K
#    Strategy: highest val score; break ties by lowest overfit gap
# ──────────────────────────────────────────────
def pick_optimal_k(results: pd.DataFrame) -> int:
    best_val  = results["val_mean"].max()
    threshold = best_val - 0.001       # within 0.1% of best is "equivalent"
    candidates = results[results["val_mean"] >= threshold]

    # Among equivalent-performance Ks, prefer smallest overfit gap
    optimal_row = candidates.loc[candidates["overfit_gap"].idxmin()]
    optimal_k   = int(optimal_row["k"])

    log.info(f"\n  Best val {SCORING}   : {best_val:.4f}")
    log.info(f"  Optimal K selected : {optimal_k}  "
             f"(val={optimal_row['val_mean']:.4f}, "
             f"gap={optimal_row['overfit_gap']:.4f})")
    return optimal_k


# ──────────────────────────────────────────────
# 4. Plot elbow curve
# ──────────────────────────────────────────────
def plot_k_curve(results: pd.DataFrame, optimal_k: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(results["k"], results["val_mean"],
            marker='o', color='#2ECC71', linewidth=2, label=f'CV val {SCORING}')
    ax.fill_between(results["k"],
                    results["val_mean"] - results["val_std"],
                    results["val_mean"] + results["val_std"],
                    alpha=0.15, color='#2ECC71', label='± 1 std')
    ax.plot(results["k"], results["train_mean"],
            marker='s', color='#3498DB', linewidth=2,
            linestyle='--', label='Train score')

    # Mark optimal K
    opt_row = results[results["k"] == optimal_k].iloc[0]
    ax.axvline(optimal_k, color='red', linestyle=':', linewidth=1.5)
    ax.scatter([optimal_k], [opt_row["val_mean"]],
               color='red', zorder=5, s=100,
               label=f'Optimal K={optimal_k}')

    ax.set_xlabel('K (number of features selected)', fontsize=12)
    ax.set_ylabel(SCORING, fontsize=12)
    ax.set_title('SelectKBest — Optimal K via Cross-Validation\n'
                 '(fitted on training set only)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  K curve saved → {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("PlasmoFAB — Optimal K Selection")
    log.info("=" * 55)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, cols = load_train()
    X_train, cols          = prefilter(X_train, cols)

    results   = search_k(X_train, y_train, max_features=len(cols))
    optimal_k = pick_optimal_k(results)

    # Save results
    results.to_csv(RESULTS_DIR / "optimal_k_search.csv", index=False)
    log.info(f"\n  Full results → {RESULTS_DIR / 'optimal_k_search.csv'}")

    # Save optimal K to file so 01_feature_selection.py can read it
    (RESULTS_DIR / "optimal_k.txt").write_text(str(optimal_k))
    log.info(f"  Optimal K    → {RESULTS_DIR / 'optimal_k.txt'}")

    plot_k_curve(results, optimal_k,
                 PLOTS_DIR / "optimal_k_curve.png")

    log.info("\n" + "=" * 55)
    log.info(f"  → Set K_BEST = {optimal_k} in 01_feature_selection.py")
    log.info("=" * 55)
