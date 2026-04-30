#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, matthews_corrcoef, roc_auc_score,
                               confusion_matrix, roc_curve, auc, log_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SPLITS_DIR   = Path("splits/selected")
RESULTS_DIR  = Path("results/cv")
PLOTS_DIR    = Path("plots/cv")
CV_FOLDS     = 10
RANDOM_STATE = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"cv_oof_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)

MODELS = {
    "SVM":      SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
    "RF":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost":  XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    "LR":       LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "NB":       GaussianNB(),
    "DT":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE),
    "QDA":      QuadraticDiscriminantAnalysis(reg_param=0.5),
    "LDA":      LinearDiscriminantAnalysis(),
    "ET":       ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GB":       GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
    "kNN":      KNeighborsClassifier(n_neighbors=5),
}

METRICS = ["accuracy", "precision", "specificity", "recall", "f1", "mcc", "roc_auc", "log_loss"]

# ──────────────────────────────────────────────
# 1. Load splits (Pooling for OOF)
# ──────────────────────────────────────────────
def load_all_data():
    log.info(f"Pooling data from '{SPLITS_DIR}' for OOF analysis...")
    train_df = pd.read_csv(SPLITS_DIR / "train_selected.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "validation_selected.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test_selected.csv")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    X = full_df.drop(columns=["label"]).values
    y = full_df["label"].values

    log.info(f"  Total samples: {len(y)} | Pos: {y.sum()} | Neg: {(y==0).sum()}")
    return X, y

# ──────────────────────────────────────────────
# 2. Metrics Helper
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_proba) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "mcc":         matthews_corrcoef(y_true, y_pred),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "log_loss":    log_loss(y_true, y_proba),
        "cm":          cm
    }

# ──────────────────────────────────────────────
# 3. Main Loop
# ──────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_all_data()
    
    all_fold_rows = []
    oof_predictions_master = pd.DataFrame({"true_label": y})
    aggregate_results = []

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in MODELS.items():
        log.info(f"\nProcessing Model: {name}")
        model_plot_dir = PLOTS_DIR / name
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        # Arrays to store OOF predictions for this specific model
        oof_preds = np.zeros(len(y))
        oof_probas = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            clf.fit(X_tr, y_tr)
            
            # Predict on the "unseen" fold
            fold_preds = clf.predict(X_val)
            fold_probas = clf.predict_proba(X_val)[:, 1]

            # Store in OOF arrays
            oof_preds[val_idx] = fold_preds
            oof_probas[val_idx] = fold_probas

            # Calculate individual fold metrics
            metrics = compute_metrics(y_val, fold_preds, fold_probas)
            row = {"model": name, "fold": fold}
            row.update({k: v for k, v in metrics.items() if k != "cm"})
            all_fold_rows.append(row)

        # After 10 folds, we have OOF predictions for every sample
        oof_predictions_master[f"{name}_pred"] = oof_preds
        oof_predictions_master[f"{name}_proba"] = oof_probas

        # Calculate Aggregate (OOF) Metrics
        agg_metrics = compute_metrics(y, oof_preds, oof_probas)
        agg_row = {"model": name}
        agg_row.update({k: v for k, v in agg_metrics.items() if k != "cm"})
        aggregate_results.append(agg_row)

        log.info(f"  OOF Aggregate AUC: {agg_metrics['roc_auc']:.4f}")

    # ── Save OOF Predictions ──
    oof_predictions_master.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    
    # ── Save Aggregate OOF Metrics ──
    pd.DataFrame(aggregate_results).sort_values("roc_auc", ascending=False).to_csv(
        RESULTS_DIR / "oof_aggregate_metrics.csv", index=False
    )

    # ── Save CV Stats (Mean/Std) ──
    cv_df = pd.DataFrame(all_fold_rows)
    summary_rows = []
    for model_name, group in cv_df.groupby("model"):
        row = {"model": model_name}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"]  = group[metric].std()
        summary_rows.append(row)
    
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "cv_summary.csv", index=False)

    log.info(f"\nCV Pipeline Complete. Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
