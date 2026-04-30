#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef, roc_auc_score,
                              confusion_matrix, roc_curve, auc, log_loss)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SPLITS_DIR  = Path("splits/selected")
PLOTS_DIR   = Path("plots")
RESULTS_DIR = Path("results")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"training_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────
MODELS = {
    "SVM":       SVC(kernel='linear', probability=True, random_state=0),
    "RF":        RandomForestClassifier(n_estimators=100, random_state=0),
    "XGBoost":   XGBClassifier(random_state=0, eval_metric='logloss', verbosity=0),
    "LR":        LogisticRegression(max_iter=1000, random_state=0),
    "NB":        GaussianNB(),
    "DT":        DecisionTreeClassifier(random_state=0),
    "AdaBoost":  AdaBoostClassifier(n_estimators=50, random_state=0),
    "QDA":       QuadraticDiscriminantAnalysis(reg_param=0.5),
    "LDA":       LinearDiscriminantAnalysis(),
    "ET":        ExtraTreesClassifier(n_estimators=100, random_state=0),
    "GB":        GradientBoostingClassifier(n_estimators=100, random_state=0),
    "LightGBM":  LGBMClassifier(random_state=0, verbosity=-1),
    "kNN":       KNeighborsClassifier(n_neighbors=5),
}


# ──────────────────────────────────────────────
# 1. Load splits
# ──────────────────────────────────────────────
def load_splits() -> dict:
    log.info(f"Loading splits from '{SPLITS_DIR}' ...")
    data = {}
    for split in ["train", "validation", "test"]:
        df = pd.read_csv(SPLITS_DIR / f"{split}_selected.csv")
        data[f"y_{split}"] = df["label"].values
        data[f"x_{split}"] = df.drop(columns=["label"]).values
        log.info(f"  {split:<12}: {data[f'x_{split}'].shape[0]} samples, "
                 f"{data[f'x_{split}'].shape[1]} features")
    return data


# ──────────────────────────────────────────────
# 2. Evaluate a single model on a split
# ──────────────────────────────────────────────
def evaluate_model(clf, X, y, split_name: str) -> dict:
    y_pred  = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]
    cm      = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "split":       split_name,
        "accuracy":    accuracy_score(y, y_pred),
        "precision":   precision_score(y, y_pred, zero_division=0),
        "recall":      recall_score(y, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1":          f1_score(y, y_pred, zero_division=0),
        "mcc":         matthews_corrcoef(y, y_pred),
        "roc_auc":     roc_auc_score(y, y_proba),
        "log_loss":    log_loss(y, clf.predict_proba(X)),
        "cm":          cm,
        "y_pred":      y_pred,
        "y_proba":     y_proba,
    }


# ──────────────────────────────────────────────
# 3. Plots
# ──────────────────────────────────────────────
def plot_roc(clf, name: str, X_test, y_test, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2,
            linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {name}', fontsize=13, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_roc.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    classes    = ['Non-Antigen', 'Antigen']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_loss_curve(name: str, train_loss: float, val_loss: float,
                    test_loss: float, out_dir: Path):
    splits = ['Train', 'Validation', 'Test']
    losses = [train_loss, val_loss, test_loss]
    colors = ['#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(splits, losses, color=colors,
                  width=0.5, edgecolor='black', linewidth=0.7)

    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)

    ax.plot(splits, losses, color='black', marker='o',
            linewidth=1.5, markersize=6, zorder=5)
    ax.set_ylabel('Log Loss', fontsize=12)
    ax.set_title(f'Train / Validation / Test Loss — {name}',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(losses) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_loss.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
# 4. Print metrics table
# ──────────────────────────────────────────────
def print_metrics(name: str, results: dict):
    log.info(f"\n{'─'*50}")
    log.info(f"  {name} — {results['split'].upper()} SET METRICS")
    log.info(f"{'─'*50}")
    for metric in ["accuracy", "precision", "specificity",
                   "recall", "f1", "mcc", "roc_auc", "log_loss"]:
        log.info(f"  {metric:<14}: {results[metric]:.4f}")
    log.info(f"  Confusion Matrix:\n{results['cm']}")


# ──────────────────────────────────────────────
# 5. Main pipeline
# ──────────────────────────────────────────────
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data    = load_splits()
    x_train = data["x_train"];     y_train = data["y_train"]
    x_val   = data["x_validation"]; y_val  = data["y_validation"]
    x_test  = data["x_test"];      y_test  = data["y_test"]

    all_results = []

    for name, clf in MODELS.items():
        model_plot_dir = PLOTS_DIR / name
        model_plot_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*50}")
        log.info(f"  Training: {name}")
        log.info(f"{'='*50}")

        try:
            # Train
            clf.fit(x_train, y_train)

            # Evaluate all three splits
            train_results = evaluate_model(clf, x_train, y_train, "train")
            val_results   = evaluate_model(clf, x_val,   y_val,   "validation")
            test_results  = evaluate_model(clf, x_test,  y_test,  "test")

            print_metrics(name, train_results)
            print_metrics(name, val_results)
            print_metrics(name, test_results)

            # Plots
            plot_roc(clf, name, x_test, y_test, model_plot_dir)
            plot_confusion_matrix(test_results["cm"], name, model_plot_dir)
            plot_loss_curve(
                name,
                train_loss = train_results["log_loss"],
                val_loss   = val_results["log_loss"],
                test_loss  = test_results["log_loss"],
                out_dir    = model_plot_dir
            )
            log.info(f"  Plots saved to '{model_plot_dir}/'")

            # Collect results
            for res in [train_results, val_results, test_results]:
                all_results.append({
                    "model":       name,
                    "split":       res["split"],
                    "accuracy":    res["accuracy"],
                    "precision":   res["precision"],
                    "specificity": res["specificity"],
                    "recall":      res["recall"],
                    "f1":          res["f1"],
                    "mcc":         res["mcc"],
                    "roc_auc":     res["roc_auc"],
                    "log_loss":    res["log_loss"],
                })

        except Exception as e:
            log.error(f"  {name} FAILED: {e}")
            continue

    # Save all results
    results_df   = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "all_model_metrics.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"\nAll metrics saved to '{results_path}'")

    # Print test set summary
    log.info("\n" + "="*75)
    log.info("SUMMARY — TEST SET")
    log.info("="*75)
    test_summary = (results_df[results_df["split"] == "test"]
                    .drop(columns="split")
                    .sort_values("roc_auc", ascending=False))
    log.info("\n" + test_summary.to_string(index=False))
    log.info("="*75)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
