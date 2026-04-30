#!/usr/bin/env python3


import sys
import logging
import warnings
import os
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    roc_curve, auc, log_loss
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_FILE   = "PlasmoFAB_seq.csv"
PLOTS_DIR    = Path("plots/deep_learning")
RESULTS_DIR  = Path("results")
MODELS_DIR   = Path("results/deep_learning_models")

RANDOM_STATE = 0          # must match 00_preprocessing_feature_extraction.py
MAX_SEQ_LEN  = 1024       # sequences truncated/padded to this length
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
PATIENCE     = 10         # early-stopping patience (epochs without val_acc improvement)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"deep_learning_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  1. DATA LOADING & SPLITTING
# ══════════════════════════════════════════════

def load_data(filepath: str) -> tuple[list, list]:
    """Load sequences + labels from PlasmoFAB CSV (same logic as script 00)."""
    log.info(f"Loading data from '{filepath}' ...")
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    sequences, labels = [], []
    with open(filepath, 'r') as f:
        next(f)   # skip header
        for i, line in enumerate(f, start=2):
            parts = line.strip().split(',')
            if len(parts) < 3:
                log.warning(f"  Skipping malformed row {i}: {line.strip()}")
                continue
            sequences.append(parts[1].strip())
            labels.append(int(parts[2][0]))

    log.info(f"  Loaded {len(sequences)} sequences | "
             f"Positives: {sum(labels)} | Negatives: {len(labels) - sum(labels)}")
    return sequences, labels


def preprocess(sequences: list, labels: list) -> tuple[list, list]:
    """Remove duplicates and unnatural AAs — identical to script 00."""
    log.info("Preprocessing sequences ...")
    seen = set()
    seqs_dedup, labels_dedup = [], []
    for seq, lbl in zip(sequences, labels):
        if seq not in seen:
            seen.add(seq)
            seqs_dedup.append(seq)
            labels_dedup.append(lbl)
    log.info(f"  Duplicates removed: {len(sequences) - len(seqs_dedup)}")

    NATURAL = set("ACDEFGHIKLMNPQRSTVWY")
    seqs_clean, labels_clean = [], []
    for seq, lbl in zip(seqs_dedup, labels_dedup):
        if all(aa in NATURAL for aa in seq.strip().upper()):
            seqs_clean.append(seq.strip().upper())
            labels_clean.append(lbl)
    log.info(f"  Unnatural sequences removed: "
             f"{len(seqs_dedup) - len(seqs_clean)}")
    log.info(f"  Sequences after preprocessing: {len(seqs_clean)}")
    return seqs_clean, labels_clean


def make_splits(sequences: list, labels: list) -> dict:
    """
    70 / 20 / 10 stratified split with random_state=0,
    identical to 00_preprocessing_feature_extraction.py.
    """
    log.info("Splitting data (70% train | 20% val | 10% test) ...")
    X = np.array(sequences)
    y = np.array(labels)

    x_temp, x_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10,
        random_state=RANDOM_STATE, stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.2222,
        random_state=RANDOM_STATE, stratify=y_temp)

    for split, (xs, ys) in [("train",      (x_train, y_train)),
                             ("validation", (x_val,   y_val)),
                             ("test",       (x_test,  y_test))]:
        log.info(f"  {split:<12}: {len(xs):>4} samples | "
                 f"pos={ys.sum()} neg={(ys==0).sum()}")

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val":   x_val,   "y_val":   y_val,
        "x_test":  x_test,  "y_test":  y_test,
    }


# ══════════════════════════════════════════════
#  2. PYTORCH DATASET
# ══════════════════════════════════════════════

AA_VOCAB = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    '<PAD>': 0,
}
VOCAB_SIZE = len(AA_VOCAB)   # 21


class ProteinDataset(Dataset):
    """Encode protein sequences as integer token tensors."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray,
                 max_len: int = MAX_SEQ_LEN):
        self.sequences = sequences
        self.labels    = labels
        self.max_len   = max_len

    def __len__(self):
        return len(self.sequences)

    def _encode(self, seq: str) -> list:
        encoded = [AA_VOCAB.get(aa, 0) for aa in seq[:self.max_len]]
        if len(encoded) < self.max_len:
            encoded += [0] * (self.max_len - len(encoded))
        return encoded

    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self._encode(self.sequences[idx]),
                                     dtype=torch.long),
            'label':    torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════
#  3. MODEL ARCHITECTURES
# ══════════════════════════════════════════════

class MLPModel(nn.Module):
    """Multi-Layer Perceptron — flattens the embedding."""
    def __init__(self, max_len: int = MAX_SEQ_LEN,
                 embed_dim: int = 32,
                 hidden_sizes=(256, 128, 64), dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        in_feat = max_len * embed_dim
        layers  = []
        for h in hidden_sizes:
            layers += [nn.Linear(in_feat, h), nn.ReLU(),
                       nn.Dropout(dropout), nn.BatchNorm1d(h)]
            in_feat = h
        layers.append(nn.Linear(in_feat, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        emb = self.embedding(x)                  # (B, L, E)
        return self.net(emb.view(emb.size(0), -1))


class CNNModel(nn.Module):
    """1-D Convolutional Neural Network with multi-scale filters."""
    def __init__(self, embed_dim: int = 128,
                 num_filters: int = 256,
                 filter_sizes=(3, 5, 7), dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 2)

    def forward(self, x):
        emb  = self.embedding(x).permute(0, 2, 1)   # (B, E, L)
        pool = [F.max_pool1d(F.relu(c(emb)),
                             F.relu(c(emb)).size(2)).squeeze(2)
                for c in self.convs]
        return self.fc(self.dropout(torch.cat(pool, dim=1)))


class RNNModel(nn.Module):
    """Vanilla RNN."""
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, h = self.rnn(self.embedding(x))
        return self.fc(self.dropout(h[-1]))


class LSTMModel(nn.Module):
    """Long Short-Term Memory."""
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(self.embedding(x))
        return self.fc(self.dropout(h[-1]))


class GRUModel(nn.Module):
    """Gated Recurrent Unit."""
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return self.fc(self.dropout(h[-1]))


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM."""
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=dropout,
                              bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        _, (h, _) = self.bilstm(self.embedding(x))
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))


class BiGRUModel(nn.Module):
    """Bidirectional GRU."""
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        _, h = self.bigru(self.embedding(x))
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))


class TransformerModel(nn.Module):
    """BERT-like Transformer encoder."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 8,
                 num_layers: int = 4, dim_ff: int = 512,
                 max_len: int = MAX_SEQ_LEN, dropout: float = 0.3):
        super().__init__()
        self.embedding   = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.pos_enc     = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(embed_dim, 2)

    def forward(self, x):
        emb = self.embedding(x) + self.pos_enc[:, :x.size(1), :]
        pad_mask = (x == 0)
        out = self.transformer(emb, src_key_padding_mask=pad_mask)
        return self.fc(self.dropout(out.mean(dim=1)))


class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1):
        super().__init__()
        pad = ks // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, ks, stride, pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, ks, 1, pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride),
                                    nn.BatchNorm1d(out_ch))
                      if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResNetModel(nn.Module):
    """1-D ResNet for protein sequences."""
    def __init__(self, embed_dim: int = 128,
                 channels=(64, 128, 256), blocks=(2, 2, 2),
                 dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, channels[0], 7, stride=2, padding=3),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers, in_ch = [], channels[0]
        for i, (n, out_ch) in enumerate(zip(blocks, channels)):
            for j in range(n):
                stride = 2 if j == 0 and i > 0 else 1
                layers.append(_ResBlock(in_ch, out_ch, stride=stride))
                in_ch = out_ch
        self.body    = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(channels[-1], 2)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)   # (B, E, L)
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))


# ══════════════════════════════════════════════
#  4. MODEL REGISTRY
# ══════════════════════════════════════════════

def build_model_registry(max_len: int) -> dict:
    return {
        "MLP":         lambda: MLPModel(max_len=max_len),
        "CNN":         lambda: CNNModel(),
        "RNN":         lambda: RNNModel(),
        "LSTM":        lambda: LSTMModel(),
        "GRU":         lambda: GRUModel(),
        "Bi-LSTM":     lambda: BiLSTMModel(),
        "Bi-GRU":      lambda: BiGRUModel(),
        "Transformer": lambda: TransformerModel(max_len=max_len),
        "ResNet":      lambda: ResNetModel(),
    }


# ══════════════════════════════════════════════
#  5. TRAINING & EVALUATION
# ══════════════════════════════════════════════

def train_one_model(model: nn.Module, model_name: str,
                    train_ds: ProteinDataset, val_ds: ProteinDataset,
                    device: torch.device) -> tuple[nn.Module, dict]:
    """Train with early stopping; return best-val model + history."""
    log.info(f"  Training {model_name} ...")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    best_val_acc = -1.0
    patience_ctr = 0
    best_state   = None

    for epoch in range(EPOCHS):
        # ── train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in train_loader:
            seqs   = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            out  = model(seqs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += out.argmax(1).eq(labels).sum().item()
            t_total   += labels.size(0)

        t_loss /= len(train_loader)
        t_acc   = t_correct / t_total

        # ── validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                seqs   = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                out    = model(seqs)
                v_loss    += criterion(out, labels).item()
                v_correct += out.argmax(1).eq(labels).sum().item()
                v_total   += labels.size(0)

        v_loss /= len(val_loader)
        v_acc   = v_correct / v_total

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        if (epoch + 1) % 5 == 0:
            log.info(f"    Epoch {epoch+1:>3}/{EPOCHS} | "
                     f"train loss={t_loss:.4f} acc={t_acc:.4f} | "
                     f"val loss={v_loss:.4f} acc={v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_ctr = 0
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"    Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    model.load_state_dict(best_state)
    return model, history


def _run_inference(model: nn.Module, dataset: ProteinDataset,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_pred, y_proba) for a dataset."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)
    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for batch in loader:
            seqs = batch['sequence'].to(device)
            out  = model(seqs)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred)
            probas.extend(prob)
    return np.array(preds), np.array(probas)


def compute_metrics(y_true, y_pred, y_proba, split: str) -> dict:
    """Same metric set as 01_train_evaluate_models.py."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # log_loss needs full probability matrix; reconstruct from p(positive)
    proba_2d = np.column_stack([1 - y_proba, y_proba])
    return {
        "split":       split,
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "mcc":         matthews_corrcoef(y_true, y_pred),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "log_loss":    log_loss(y_true, proba_2d),
        "cm":          cm,
        "y_pred":      y_pred,
        "y_proba":     y_proba,
    }


# ══════════════════════════════════════════════
#  6. PLOTS  (same style as 01_train_evaluate_models.py)
# ══════════════════════════════════════════════

def plot_roc(name: str, y_test, y_proba, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_val     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_val:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2,
            linestyle='--', label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_roc.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    classes = ['Non-Antigen', 'Antigen']
    ticks   = np.arange(2)
    ax.set_xticks(ticks); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=13,
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_loss_curve(name: str, train_loss: float, val_loss: float,
                    test_loss: float, out_dir: Path):
    """Bar + line chart matching 01_train_evaluate_models.py style."""
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
    ax.set_ylim(0, max(losses) * 1.35)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_loss.png", dpi=150)
    plt.close(fig)


def plot_training_history(name: str, history: dict, out_dir: Path):
    """Epoch-level loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in [
        (axes[0], ('train_loss', 'val_loss'), 'Loss'),
        (axes[1], ('train_acc',  'val_acc'),  'Accuracy'),
    ]:
        ax.plot(history[metric[0]], label='Train',      color='#4C72B0', lw=2)
        ax.plot(history[metric[1]], label='Validation', color='#DD8452', lw=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{name} — {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_training_history.png", dpi=150)
    plt.close(fig)


def plot_all_roc(predictions: dict, y_test, out_dir: Path):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, preds in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
        roc_val     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Deep Learning Models',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "dl_roc_all.png", dpi=150)
    plt.close(fig)
    log.info(f"  All-model ROC saved -> {out_dir / 'dl_roc_all.png'}")


def plot_performance_comparison(test_metrics: dict, out_dir: Path):
    """Grouped bar chart comparing all models on test-set metrics."""
    metric_keys = ['accuracy', 'precision', 'specificity',
                   'recall', 'f1', 'mcc', 'roc_auc']
    models  = list(test_metrics.keys())
    x       = np.arange(len(models))
    width   = 0.11
    colors  = plt.cm.tab10(np.linspace(0, 1, len(metric_keys)))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (mk, color) in enumerate(zip(metric_keys, colors)):
        vals = [test_metrics[m][mk] for m in models]
        ax.bar(x + i * width, vals, width,
               label=mk.replace('_', ' ').title(), color=color)

    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('Performance Metrics — Deep Learning Models (Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "dl_performance_comparison.png", dpi=150)
    plt.close(fig)
    log.info(f"  Performance comparison saved -> "
             f"{out_dir / 'dl_performance_comparison.png'}")


# ══════════════════════════════════════════════
#  7. PDF REPORT
# ══════════════════════════════════════════════

def generate_pdf_report(test_summary_df: pd.DataFrame,
                        out_dir: Path, plot_dir: Path):
    pdf_path = out_dir / "Deep_Learning_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        # ── Title page ──
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.75, 'Deep Learning Models for Antigen Prediction',
                 ha='center', fontsize=22, fontweight='bold')
        fig.text(0.5, 0.65,
                 'Plasmodium falciparum — PlasmoFAB Dataset',
                 ha='center', fontsize=16)
        fig.text(0.5, 0.55,
                 f'Date: {datetime.now():%B %d, %Y}',
                 ha='center', fontsize=13)
        model_list = '\n'.join(
            [f'• {m}' for m in test_summary_df.index])
        fig.text(0.5, 0.35, 'Models Evaluated:\n' + model_list,
                 ha='center', fontsize=11)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ── Results table ──
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        display = test_summary_df.round(4)
        table = ax.table(
            cellText=display.values,
            colLabels=display.columns,
            rowLabels=display.index,
            cellLoc='center', loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        for j in range(len(display.columns)):
            table[(0, j)].set_facecolor('#2C3E50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        ax.set_title('Test Set Performance Metrics',
                     fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # ── Embed saved plots ──
        for plot_name in ['dl_roc_all', 'dl_performance_comparison']:
            path = plot_dir / f"{plot_name}.png"
            if path.exists():
                from PIL import Image
                img = Image.open(path)
                fig = plt.figure(figsize=(11, 8.5))
                plt.imshow(img); plt.axis('off')
                pdf.savefig(fig, bbox_inches='tight'); plt.close()

    log.info(f"  PDF report saved -> {pdf_path}")
    return pdf_path


# ══════════════════════════════════════════════
#  8. MAIN
# ══════════════════════════════════════════════

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PlasmoFAB — Deep Learning Pipeline")
    log.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Device: {device}")

    # ── Data ──
    sequences, labels = load_data(INPUT_FILE)
    sequences, labels = preprocess(sequences, labels)
    splits = make_splits(sequences, labels)

    max_len = min(max(len(s) for s in sequences), MAX_SEQ_LEN)
    log.info(f"  Max sequence length (capped): {max_len}")

    train_ds = ProteinDataset(splits['x_train'], splits['y_train'], max_len)
    val_ds   = ProteinDataset(splits['x_val'],   splits['y_val'],   max_len)
    test_ds  = ProteinDataset(splits['x_test'],  splits['y_test'],  max_len)

    MODEL_REGISTRY = build_model_registry(max_len)

    all_results  = []          # rows for deep_learning_metrics.csv
    predictions  = {}          # for the combined ROC plot
    test_metrics = {}          # for the performance bar chart
    history_rows = []          # for training history CSV

    for name, build_fn in MODEL_REGISTRY.items():
        log.info(f"\n{'='*60}")
        log.info(f"  Model: {name}")
        log.info(f"{'='*60}")

        try:
            model = build_fn()
            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
            log.info(f"  Parameters: {n_params:,}")

            model, history = train_one_model(
                model, name, train_ds, val_ds, device)

            # Save model weights
            torch.save(model.state_dict(),
                       MODELS_DIR / f"{name}_best.pth")

            # Evaluate all three splits
            y_train_true = splits['y_train']
            y_val_true   = splits['y_val']
            y_test_true  = splits['y_test']

            tr_pred, tr_prob = _run_inference(model, train_ds, device)
            va_pred, va_prob = _run_inference(model, val_ds,   device)
            te_pred, te_prob = _run_inference(model, test_ds,  device)

            train_res = compute_metrics(y_train_true, tr_pred, tr_prob, "train")
            val_res   = compute_metrics(y_val_true,   va_pred, va_prob, "validation")
            test_res  = compute_metrics(y_test_true,  te_pred, te_prob, "test")

            # Log test metrics
            log.info(f"\n  ── Test set results ──────────────────────")
            for k in ['accuracy', 'precision', 'specificity',
                      'recall', 'f1', 'mcc', 'roc_auc', 'log_loss']:
                log.info(f"  {k:<14}: {test_res[k]:.4f}")
            log.info(f"  Confusion Matrix:\n{test_res['cm']}")

            # Plots
            model_dir = PLOTS_DIR / name
            model_dir.mkdir(parents=True, exist_ok=True)

            plot_roc(name, y_test_true, te_prob, model_dir)
            plot_confusion_matrix(test_res['cm'], name, model_dir)
            plot_loss_curve(name,
                            train_res['log_loss'],
                            val_res['log_loss'],
                            test_res['log_loss'], model_dir)
            plot_training_history(name, history, model_dir)
            log.info(f"  Plots saved to '{model_dir}/'")

            # Collect results (same schema as all_model_metrics.csv)
            for res in [train_res, val_res, test_res]:
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

            # Collect for summary plots
            predictions[name]  = {'y_proba': te_prob, 'y_pred': te_pred}
            test_metrics[name] = test_res

            # Training history rows
            for epoch_idx, (tl, ta, vl, va) in enumerate(zip(
                    history['train_loss'], history['train_acc'],
                    history['val_loss'],  history['val_acc'])):
                history_rows.append({
                    'model': name, 'epoch': epoch_idx + 1,
                    'train_loss': tl, 'train_acc': ta,
                    'val_loss': vl,   'val_acc': va,
                })

        except Exception as e:
            log.error(f"  {name} FAILED: {e}", exc_info=True)
            continue

    # ── Summary plots ──
    log.info("\nGenerating summary plots ...")
    if predictions:
        plot_all_roc(predictions, splits['y_test'], PLOTS_DIR)
        plot_performance_comparison(test_metrics, PLOTS_DIR)

    # ── Save CSVs ──
    metrics_df = pd.DataFrame(all_results)
    metrics_path = RESULTS_DIR / "deep_learning_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"  All metrics saved -> {metrics_path}")

    hist_df = pd.DataFrame(history_rows)
    hist_path = RESULTS_DIR / "deep_learning_training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    log.info(f"  Training history saved -> {hist_path}")

    # ── Test set summary table ──
    metric_cols = ['accuracy', 'precision', 'specificity',
                   'recall', 'f1', 'mcc', 'roc_auc', 'log_loss']
    test_summary = (metrics_df[metrics_df["split"] == "test"]
                    .set_index("model")
                    .drop(columns="split")[metric_cols]
                    .sort_values("roc_auc", ascending=False))

    log.info("\n" + "=" * 75)
    log.info("SUMMARY — TEST SET")
    log.info("=" * 75)
    log.info("\n" + test_summary.round(4).to_string())
    log.info("=" * 75)

    # ── PDF report ──
    if predictions:
        generate_pdf_report(test_summary, RESULTS_DIR, PLOTS_DIR)

    log.info("Deep learning pipeline complete.")


if __name__ == "__main__":
    main()
