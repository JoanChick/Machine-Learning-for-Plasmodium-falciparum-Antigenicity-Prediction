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

from sklearn.model_selection import StratifiedKFold, train_test_split
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

TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
PLOTS_DIR    = Path(f"plots/dl_cv_{TIMESTAMP}")
RESULTS_DIR  = Path(f"results/dl_cv_{TIMESTAMP}")
MODELS_DIR   = Path(f"results/dl_cv_{TIMESTAMP}/models")

N_SPLITS     = 10         # 10-Fold CV
RANDOM_STATE = 0          
MAX_SEQ_LEN  = 1024       
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
PATIENCE     = 10         

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"dl_cv_{TIMESTAMP}.log")
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════
def load_data(filepath: str) -> tuple[list, list]:
    log.info(f"Loading data from '{filepath}' ...")
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    sequences, labels = [], []
    with open(filepath, 'r') as f:
        next(f)   
        for i, line in enumerate(f, start=2):
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            sequences.append(parts[1].strip())
            labels.append(int(parts[2][0]))

    log.info(f"  Loaded {len(sequences)} sequences | Positives: {sum(labels)} | Negatives: {len(labels) - sum(labels)}")
    return sequences, labels

def preprocess(sequences: list, labels: list) -> tuple[list, list]:
    log.info("Preprocessing sequences ...")
    seen = set()
    seqs_dedup, labels_dedup = [], []
    for seq, lbl in zip(sequences, labels):
        if seq not in seen:
            seen.add(seq)
            seqs_dedup.append(seq)
            labels_dedup.append(lbl)
    
    NATURAL = set("ACDEFGHIKLMNPQRSTVWY")
    seqs_clean, labels_clean = [], []
    for seq, lbl in zip(seqs_dedup, labels_dedup):
        if all(aa in NATURAL for aa in seq.strip().upper()):
            seqs_clean.append(seq.strip().upper())
            labels_clean.append(lbl)
            
    log.info(f"  Sequences after preprocessing: {len(seqs_clean)}")
    return seqs_clean, labels_clean


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
VOCAB_SIZE = len(AA_VOCAB)

class ProteinDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, max_len: int = MAX_SEQ_LEN):
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
            'sequence': torch.tensor(self._encode(self.sequences[idx]), dtype=torch.long),
            'label':    torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════
#  3. MODEL ARCHITECTURES
# ══════════════════════════════════════════════
class MLPModel(nn.Module):
    def __init__(self, max_len: int = MAX_SEQ_LEN, embed_dim: int = 32, hidden_sizes=(256, 128, 64), dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        in_feat = max_len * embed_dim
        layers  = []
        for h in hidden_sizes:
            layers += [nn.Linear(in_feat, h), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(h)]
            in_feat = h
        layers.append(nn.Linear(in_feat, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        emb = self.embedding(x)
        return self.net(emb.view(emb.size(0), -1))

class CNNModel(nn.Module):
    def __init__(self, embed_dim: int = 128, num_filters: int = 256, filter_sizes=(3, 5, 7), dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 2)

    def forward(self, x):
        emb  = self.embedding(x).permute(0, 2, 1)
        pool = [F.max_pool1d(F.relu(c(emb)), F.relu(c(emb)).size(2)).squeeze(2) for c in self.convs]
        return self.fc(self.dropout(torch.cat(pool, dim=1)))

class RNNModel(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, h = self.rnn(self.embedding(x))
        return self.fc(self.dropout(h[-1]))

class LSTMModel(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(self.embedding(x))
        return self.fc(self.dropout(h[-1]))

class GRUModel(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, h = self.gru(self.embedding(x))
        return self.fc(self.dropout(h[-1]))

class BiLSTMModel(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        _, (h, _) = self.bilstm(self.embedding(x))
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))

class BiGRUModel(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.bigru = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        _, h = self.bigru(self.embedding(x))
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))

class TransformerModel(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 4, dim_ff: int = 512, max_len: int = MAX_SEQ_LEN, dropout: float = 0.3):
        super().__init__()
        self.embedding   = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.pos_enc     = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        enc_layer        = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
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
        self.skip  = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride), nn.BatchNorm1d(out_ch)) if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))

class ResNetModel(nn.Module):
    def __init__(self, embed_dim: int = 128, channels=(64, 128, 256), blocks=(2, 2, 2), dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.stem = nn.Sequential(nn.Conv1d(embed_dim, channels[0], 7, stride=2, padding=3), nn.BatchNorm1d(channels[0]), nn.ReLU(), nn.MaxPool1d(3, stride=2, padding=1))
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
        x = self.embedding(x).permute(0, 2, 1)
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))

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
#  4. TRAINING & EVALUATION
# ══════════════════════════════════════════════
def train_one_model(model: nn.Module, fold: int, train_ds: ProteinDataset, val_ds: ProteinDataset, device: torch.device) -> tuple[nn.Module, dict]:
    model = model.to(device)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    best_val_acc = -1.0
    patience_ctr = 0
    best_state   = None

    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in train_loader:
            seqs, labels = batch['sequence'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            out  = model(seqs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += out.argmax(1).eq(labels).sum().item()
            t_total   += labels.size(0)

        t_loss /= len(train_loader); t_acc = t_correct / t_total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                seqs, labels = batch['sequence'].to(device), batch['label'].to(device)
                out    = model(seqs)
                v_loss    += criterion(out, labels).item()
                v_correct += out.argmax(1).eq(labels).sum().item()
                v_total   += labels.size(0)

        v_loss /= len(val_loader); v_acc = v_correct / v_total

        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss);   history['val_acc'].append(v_acc)

        if (epoch + 1) % 5 == 0:
            log.info(f"    Fold {fold} - Epoch {epoch+1:>3}/{EPOCHS} | train loss={t_loss:.4f} acc={t_acc:.4f} | val loss={v_loss:.4f} acc={v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"    Fold {fold} - Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history

def _run_inference(model: nn.Module, dataset: ProteinDataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for batch in loader:
            seqs = batch['sequence'].to(device)
            out  = model(seqs)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred); probas.extend(prob)
    return np.array(preds), np.array(probas)

def compute_metrics(y_true, y_pred, y_proba, fold_str: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    proba_2d = np.column_stack([1 - y_proba, y_proba])
    return {
        "fold":        fold_str,
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "mcc":         matthews_corrcoef(y_true, y_pred),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "log_loss":    log_loss(y_true, proba_2d),
        "cm":          cm,
    }


# ══════════════════════════════════════════════
#  5. PLOTS  
# ══════════════════════════════════════════════
def _model_dir(name: str, subfolder: str = "") -> Path:
    d = PLOTS_DIR / name.replace(" ", "_").replace("/", "_")
    if subfolder:
        d = d / subfolder
    d.mkdir(parents=True, exist_ok=True)
    return d

def plot_roc(name: str, y_test, y_proba, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_val     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_val:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_roc.png", dpi=150); plt.close(fig)

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
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13, color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=150); plt.close(fig)

def plot_training_history(name: str, history: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in [(axes[0], ('train_loss', 'val_loss'), 'Loss'), (axes[1], ('train_acc',  'val_acc'),  'Accuracy')]:
        ax.plot(history[metric[0]], label='Train', color='#4C72B0', lw=2)
        ax.plot(history[metric[1]], label='Validation', color='#DD8452', lw=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{name} — {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_history.png", dpi=150); plt.close(fig)

def plot_all_roc(predictions: dict, y_test, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, preds in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Cross-Validation Aggregate ROC (OOF) — Deep Learning', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "dl_cv_roc_all.png", dpi=150); plt.close(fig)

def plot_performance_comparison(test_metrics: dict, out_dir: Path):
    metric_keys = ['accuracy', 'precision', 'specificity', 'recall', 'f1', 'mcc', 'roc_auc']
    models  = list(test_metrics.keys())
    x       = np.arange(len(models))
    width   = 0.11
    colors  = plt.cm.tab10(np.linspace(0, 1, len(metric_keys)))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (mk, color) in enumerate(zip(metric_keys, colors)):
        vals = [test_metrics[m][mk] for m in models]
        ax.bar(x + i * width, vals, width, label=mk.replace('_', ' ').title(), color=color)

    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('CV OOF Performance Metrics — Deep Learning Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "dl_cv_performance_comparison.png", dpi=150); plt.close(fig)


# ══════════════════════════════════════════════
#  6. PDF REPORT
# ══════════════════════════════════════════════
def generate_pdf_report(test_summary_df: pd.DataFrame, out_dir: Path, plot_dir: Path):
    pdf_path = out_dir / "Deep_Learning_CV_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.75, 'Deep Learning Models for Antigen Prediction\n(10-Fold Cross-Validation)', ha='center', fontsize=22, fontweight='bold')
        fig.text(0.5, 0.62, 'Plasmodium falciparum — PlasmoFAB Dataset', ha='center', fontsize=16)
        fig.text(0.5, 0.52, f'Date: {datetime.now():%B %d, %Y}', ha='center', fontsize=13)
        model_list = '\n'.join([f'• {m}' for m in test_summary_df.index])
        fig.text(0.5, 0.30, 'Models Evaluated (OOF Test Metrics):\n' + model_list, ha='center', fontsize=11)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        display = test_summary_df.round(4)
        table = ax.table(cellText=display.values, colLabels=display.columns, rowLabels=display.index, cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 2)
        for j in range(len(display.columns)):
            table[(0, j)].set_facecolor('#2C3E50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        ax.set_title('Cross-Validation Aggregate (OOF) Performance', fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        for plot_name in ['dl_cv_roc_all', 'dl_cv_performance_comparison']:
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
#  7. MAIN
# ══════════════════════════════════════════════
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PlasmoFAB — CV Deep Learning Pipeline")
    log.info(f"  CV Folds : {N_SPLITS}")
    log.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Device: {device}")

    sequences, labels = load_data(INPUT_FILE)
    sequences, labels = preprocess(sequences, labels)
    X, y = np.array(sequences), np.array(labels)

    max_len = min(max(len(s) for s in sequences), MAX_SEQ_LEN)
    log.info(f"  Max sequence length (capped): {max_len}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    MODEL_REGISTRY = build_model_registry(max_len)

    all_results  = []          
    predictions  = {}          
    test_metrics = {}          
    history_rows = []          

    for name, build_fn in MODEL_REGISTRY.items():
        log.info(f"\n{'='*60}\n  Model: {name}\n{'='*60}")
        try:
            oof_pred = np.zeros(len(X))
            oof_proba = np.zeros(len(X))
            
            for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y), 1):
                log.info(f"\n  --- {name} | Fold {fold}/{N_SPLITS} ---")
                
                # Force fresh model instantiation per fold to prevent weight leakage
                model = build_fn().to(device)
                
                tr_idx, va_idx = train_test_split(train_val_idx, test_size=0.1111, random_state=RANDOM_STATE, stratify=y[train_val_idx])

                train_ds = ProteinDataset(X[tr_idx], y[tr_idx], max_len)
                val_ds   = ProteinDataset(X[va_idx], y[va_idx], max_len)
                test_ds  = ProteinDataset(X[test_idx], y[test_idx], max_len)

                model, history = train_one_model(model, fold, train_ds, val_ds, device)
                torch.save(model.state_dict(), MODELS_DIR / f"{name}_fold_{fold}_best.pth")

                te_pred, te_prob = _run_inference(model, test_ds, device)
                oof_pred[test_idx] = te_pred
                oof_proba[test_idx] = te_prob

                fold_res = compute_metrics(y[test_idx], te_pred, te_prob, f"fold_{fold}")
                fold_res["model"] = name
                all_results.append(fold_res)

                # Save history plots for individual folds
                mdir_fold = _model_dir(name, f"fold_{fold}")
                plot_training_history(f"{name}_Fold_{fold}", history, mdir_fold)

                for epoch_idx, (tl, ta, vl, va) in enumerate(zip(history['train_loss'], history['train_acc'], history['val_loss'], history['val_acc'])):
                    history_rows.append({'model': name, 'fold': fold, 'epoch': epoch_idx + 1, 'train_loss': tl, 'train_acc': ta, 'val_loss': vl, 'val_acc': va})

            # Evaluate OOF Aggregate
            oof_m = compute_metrics(y, oof_pred, oof_proba, "OOF_Aggregate")
            oof_m["model"] = name
            all_results.append(oof_m)
            
            predictions[name]  = {'y_proba': oof_proba, 'y_pred': oof_pred}
            test_metrics[name] = oof_m

            # Plot OOF Curve for this model
            mdir_oof = _model_dir(name, "OOF_Aggregate")
            plot_roc(f"{name}_OOF", y, oof_proba, mdir_oof)
            plot_confusion_matrix(oof_m["cm"], f"{name}_OOF", mdir_oof)
            log.info(f"  OOF Plots saved -> {mdir_oof}/")

        except Exception as e:
            log.error(f"  {name} FAILED: {e}", exc_info=True)
            continue

    log.info("\nGenerating global CV summary plots ...")
    if predictions:
        plot_all_roc(predictions, y, PLOTS_DIR)
        plot_performance_comparison(test_metrics, PLOTS_DIR)

    metrics_df = pd.DataFrame(all_results)
    metrics_path = RESULTS_DIR / "dl_cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"  All metrics saved -> {metrics_path}")

    hist_df = pd.DataFrame(history_rows)
    hist_path = RESULTS_DIR / "dl_cv_training_history.csv"
    hist_df.to_csv(hist_path, index=False)
    log.info(f"  Training history saved -> {hist_path}")

    metric_cols = ['accuracy', 'precision', 'specificity', 'recall', 'f1', 'mcc', 'roc_auc', 'log_loss']
    test_summary = (metrics_df[metrics_df["fold"] == "OOF_Aggregate"]
                    .set_index("model")
                    .drop(columns="fold")[metric_cols]
                    .sort_values("roc_auc", ascending=False))

    log.info("\n" + "=" * 75)
    log.info("SUMMARY — CV OOF AGGREGATE")
    log.info("=" * 75)
    log.info("\n" + test_summary.round(4).to_string())
    log.info("=" * 75)
   
    # ── CV summary: mean ± std across folds per model ──
    summary_rows = []
    for model_name, group in cv_df.groupby("model"):
        row = {"model": model_name}
        for metric in METRICS:
            if metric in group.columns:
                row[f"{metric}_mean"] = round(group[metric].mean(), 4)
                row[f"{metric}_std"]  = round(group[metric].std(),  4)
        summary_rows.append(row)

    cv_summary = pd.DataFrame(summary_rows)\
                   .sort_values("roc_auc_mean", ascending=False)
    cv_summary.to_csv(RESULTS_DIR / "dl_cv_summary.csv", index=False)
    log.info(f"  CV summary saved       → {RESULTS_DIR / 'cv_summary.csv'}")

    # ── Save test results ──
    test_df = pd.DataFrame(all_test_rows)\
                .sort_values("roc_auc", ascending=False)
    test_df.to_csv(RESULTS_DIR / "dl_test_results.csv", index=False)
    log.info(f"  Test results saved     → {RESULTS_DIR / 'test_results.csv'}")

    if predictions:
        generate_pdf_report(test_summary, RESULTS_DIR, PLOTS_DIR)

    log.info("CV Deep Learning pipeline complete.")

if __name__ == "__main__":
    main()
