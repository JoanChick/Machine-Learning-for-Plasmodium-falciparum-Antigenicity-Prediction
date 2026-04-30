#!/usr/bin/env python3

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_DIR          = Path("splits")
OUTPUT_DIR         = Path("splits/selected")
VARIANCE_THRESH    = 0.00
CORRELATION_THRESH = 0.95
K_BEST             = 1000        # tune: 300–1000 depending on dataset size

# Feature group prefix map — matches 00_preprocessing output
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"feature_selection_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_split(name: str) -> tuple[np.ndarray, np.ndarray, list]:
    path = INPUT_DIR / f"{name}.csv"
    df   = pd.read_csv(path)
    y    = df["label"].values
    X    = df.drop(columns=["label"]).values
    cols = df.drop(columns=["label"]).columns.tolist()
    log.info(f"  Loaded {name:<12} > {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, cols


def save_split(X: np.ndarray, y: np.ndarray, cols: list, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "label", y)
    path = OUTPUT_DIR / f"{name}_selected.csv"
    df.to_csv(path, index=False)
    log.info(f"  Saved  {name:<12} > {path}")


def group_summary(cols: list, stage: str) -> pd.DataFrame:
    """Count surviving features per biological group at a given stage."""
    rows = []
    for group, prefixes in FEATURE_GROUPS.items():
        count = sum(1 for c in cols if any(c.startswith(p) for p in prefixes))
        rows.append({"group": group, stage: count})
    return pd.DataFrame(rows).set_index("group")


# ──────────────────────────────────────────────
# Step 1 — Variance Threshold
# ──────────────────────────────────────────────
def step_variance(X_train, X_val, X_test, cols):
    log.info(f"\n[Step 1] Variance Threshold (threshold={VARIANCE_THRESH})")
    n_before = len(cols)

    sel = VarianceThreshold(threshold=VARIANCE_THRESH)
    sel.fit(X_train)                        # fit on TRAIN only

    mask     = sel.get_support()
    cols_    = [c for c, m in zip(cols, mask) if m]
    X_train_ = X_train[:, mask]
    X_val_   = X_val[:, mask]
    X_test_  = X_test[:, mask]

    log.info(f"  Removed:   {n_before - len(cols_)}")
    log.info(f"  Remaining: {len(cols_)}")
    return X_train_, X_val_, X_test_, cols_


# ──────────────────────────────────────────────
# Step 2 — Correlation Filter
# ──────────────────────────────────────────────
def step_correlation(X_train, X_val, X_test, cols):
    log.info(f"\n[Step 2] Correlation Filter (threshold={CORRELATION_THRESH})")
    n_before = len(cols)

    df_train = pd.DataFrame(X_train, columns=cols)
    corr     = df_train.corr().abs()
    upper    = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # From each correlated pair, drop the one with lower mean correlation
    # (keeps the more "unique" feature from each group)
    to_drop  = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        correlated = upper.index[upper[col] > CORRELATION_THRESH].tolist()
        to_drop.update(correlated)

    keep     = [c not in to_drop for c in cols]
    cols_    = [c for c, k in zip(cols, keep) if k]
    X_train_ = df_train.loc[:, keep].values
    X_val_   = pd.DataFrame(X_val,  columns=cols).loc[:, keep].values
    X_test_  = pd.DataFrame(X_test, columns=cols).loc[:, keep].values

    log.info(f"  Removed:   {n_before - len(cols_)}")
    log.info(f"  Remaining: {len(cols_)}")
    return X_train_, X_val_, X_test_, cols_


# ──────────────────────────────────────────────
# Step 3 — SelectKBest (f_classif)
# ──────────────────────────────────────────────
def step_kbest(X_train, y_train, X_val, X_test, cols):
    k = min(K_BEST, X_train.shape[1])
    log.info(f"\n[Step 3] SelectKBest — f_classif (k={k})")
    n_before = len(cols)

    sel = SelectKBest(score_func=f_classif, k=k)
    sel.fit(X_train, y_train)               # fit on TRAIN only

    mask     = sel.get_support()
    scores   = pd.Series(sel.scores_, index=cols)

    cols_    = [c for c, m in zip(cols, mask) if m]
    X_train_ = X_train[:, mask]
    X_val_   = X_val[:, mask]
    X_test_  = X_test[:, mask]

    log.info(f"  Removed:   {n_before - len(cols_)}")
    log.info(f"  Remaining: {len(cols_)}")

    return X_train_, X_val_, X_test_, cols_, scores


# ──────────────────────────────────────────────
# Save artifacts
# ──────────────────────────────────────────────
def save_artifacts(original_n, after_var, after_corr, after_kbest,
                   cols_final, scores, group_snapshots):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Selected feature names
    (OUTPUT_DIR / "selected_feature_names.txt").write_text("\n".join(cols_final))

    # 2. F-scores for selected features
    scores_path = OUTPUT_DIR / "feature_scores.csv"
    scores[cols_final].sort_values(ascending=False)\
        .reset_index()\
        .rename(columns={"index": "feature", 0: "f_score"})\
        .to_csv(scores_path, index=False)

    # 3. Selection report
    pd.DataFrame([
        {"step": "Original",           "features": original_n},
        {"step": "Variance Threshold", "features": after_var},
        {"step": "Correlation Filter", "features": after_corr},
        {"step": "SelectKBest",        "features": after_kbest},
    ]).to_csv(OUTPUT_DIR / "selection_report.csv", index=False)

    # 4. Feature group survival summary — how many features from each
    #    biological group made it through each step
    summary = group_snapshots["original"]\
        .join(group_snapshots["after_variance"],  rsuffix="")\
        .join(group_snapshots["after_corr"],      rsuffix="")\
        .join(group_snapshots["after_kbest"],     rsuffix="")
    summary.columns = ["original", "after_variance",
                       "after_correlation", "after_kbest"]
    summary_path = OUTPUT_DIR / "feature_group_summary.csv"
    summary.to_csv(summary_path)
    log.info(f"\n  Feature group summary:\n{summary.to_string()}")
    log.info(f"\n  Saved > {summary_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("PlasmoFAB — Feature Selection (group-preserving)")
    log.info("=" * 55)

    # Load
    log.info("\nLoading splits ...")
    X_train, y_train, cols = load_split("train")
    X_val,   y_val,   _    = load_split("validation")
    X_test,  y_test,  _    = load_split("test")
    original_n = len(cols)

    group_snapshots = {"original": group_summary(cols, "original")}

    # Step 1
    X_train, X_val, X_test, cols = step_variance(X_train, X_val, X_test, cols)
    after_var = len(cols)
    group_snapshots["after_variance"] = group_summary(cols, "after_variance")

    # Step 2
    X_train, X_val, X_test, cols = step_correlation(X_train, X_val, X_test, cols)
    after_corr = len(cols)
    group_snapshots["after_corr"] = group_summary(cols, "after_corr")

    # Step 3
    X_train, X_val, X_test, cols, scores = step_kbest(
        X_train, y_train, X_val, X_test, cols)
    after_kbest = len(cols)
    group_snapshots["after_kbest"] = group_summary(cols, "after_kbest")

    # Save selected splits
    log.info("\nSaving selected splits ...")
    save_split(X_train, y_train, cols, "train")
    save_split(X_val,   y_val,   cols, "validation")
    save_split(X_test,  y_test,  cols, "test")

    # Save artifacts
    save_artifacts(original_n, after_var, after_corr, after_kbest,
                   cols, scores, group_snapshots)

    # Summary
    log.info("\n" + "=" * 55)
    log.info("REDUCTION SUMMARY")
    log.info(f"  Original features    : {original_n}")
    log.info(f"  After variance filter: {after_var}")
    log.info(f"  After corr filter    : {after_corr}")
    log.info(f"  After SelectKBest    : {after_kbest}")
    log.info(f"  Total reduction      : {original_n} > {after_kbest} "
             f"({100*(1-after_kbest/original_n):.1f}% smaller)")
    log.info("=" * 55)
    log.info("Feature selection complete.")
