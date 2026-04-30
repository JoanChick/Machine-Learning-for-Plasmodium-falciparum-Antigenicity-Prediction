#!/usr/bin/env python3

import sys
import os
import glob
import logging
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

# ── Optional HuggingFace import ──────────────────────────────────────────────
try:
    from transformers import (
        BertModel, BertTokenizer,
        T5EncoderModel, T5Tokenizer,
        EsmModel, EsmTokenizer,
        AutoModel, AutoTokenizer,
        logging as hf_logging,
    )
    hf_logging.set_verbosity_error()
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("WARNING: transformers not installed.\n"
          "  Run: pip install transformers sentencepiece")

# ── Optional ProteinBERT (Brandes et al.) ────────────────────────────────────
try:
    from proteinbert import load_pretrained_model
    PROTEINBERT_AVAILABLE = True
except ImportError:
    PROTEINBERT_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
INPUT_FILE   = "PlasmoFAB_seq.csv"

# --- RESUME LOGIC (DIRECTORY LOOKUP) ---
_existing = sorted(glob.glob("results/plm_cv_*"))
if _existing:
    TIMESTAMP = Path(_existing[-1]).name.replace("plm_cv_", "")
    print(f"Auto-resuming existing run directory: {TIMESTAMP}")
else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

PLOTS_DIR    = Path(f"plots/plm_cv_{TIMESTAMP}")
RESULTS_DIR  = Path(f"results/plm_cv_{TIMESTAMP}")
CKPT_DIR     = Path(f"results/plm_cv_{TIMESTAMP}/checkpoints")

N_SPLITS     = 10     # 10-Fold CV
RANDOM_STATE = 0      
EPOCHS       = 50     
BATCH_SIZE   = 32     
LR_HEAD      = 1e-3   
LR_PLM       = 2e-5   
PATIENCE     = 10     
FREEZE_PLM   = True   
MAX_LEN      = 1024   

# ── Models to run ─────────────────────────────────────────────────────────────
MODELS_TO_RUN = [
    "ProtBERT",
    "ProtBERT-BFD",
    "ProtT5-XL",
    "ESM-1b",
    "ESM-2-8M",
    "ESM-2-35M",
    "ESM-2-650M",
    "Ankh-base",
    "ProteinBERT",
]

# ── HuggingFace model IDs ─────────────────────────────────────────────────────
HF_IDS = {
    "ProtBERT":     ("Rostlab/prot_bert",               "bert",  512),
    "ProtBERT-BFD": ("Rostlab/prot_bert_bfd",           "bert",  512),
    "ProtT5-XL":    ("Rostlab/prot_t5_xl_uniref50",     "t5",    512),
    "ESM-1b":       ("facebook/esm1b_t33_650M_UR50S",   "esm",  1022),
    "ESM-2-8M":     ("facebook/esm2_t6_8M_UR50D",       "esm",  1022),
    "ESM-2-35M":    ("facebook/esm2_t12_35M_UR50D",     "esm",  1022),
    "ESM-2-650M":   ("facebook/esm2_t33_650M_UR50D",    "esm",  1022),
    "Ankh-base":    ("ElnaggarLab/ankh-base",            "t5",    512),
}


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"plm_cv_{TIMESTAMP}.log")
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading 
# ══════════════════════════════════════════════════════════════════════════════
def load_data(filepath: str) -> tuple[list, list]:
    log.info(f"Loading data from '{filepath}' ...")
    if not Path(filepath).exists():
        raise FileNotFoundError(filepath)
    sequences, labels = [], []
    with open(filepath) as f:
        next(f)
        for i, line in enumerate(f, 2):
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            sequences.append(parts[1].strip())
            labels.append(int(parts[2][0]))
    log.info(f"  Loaded {len(sequences)} sequences | pos={sum(labels)}  neg={len(labels)-sum(labels)}")
    return sequences, labels

def preprocess(sequences: list, labels: list) -> tuple[list, list]:
    NATURAL = set("ACDEFGHIKLMNPQRSTVWY")
    seen, seqs_out, labs_out = set(), [], []
    for seq, lbl in zip(sequences, labels):
        s = seq.strip().upper()
        if s not in seen and all(aa in NATURAL for aa in s):
            seen.add(s); seqs_out.append(s); labs_out.append(lbl)
    log.info(f"  After dedup + natural-AA filter: {len(seqs_out)} sequences")
    return seqs_out, labs_out


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sequence Formatters
# ══════════════════════════════════════════════════════════════════════════════
def fmt_rostlab(seq: str) -> str:
    for rare in "UZOB":
        seq = seq.replace(rare, "X")
    return " ".join(seq)

def fmt_standard(seq: str) -> str:
    return seq

SEQ_FORMATTERS = {
    "bert": fmt_rostlab,
    "t5":   fmt_rostlab,
    "esm":  fmt_standard,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLM Dataset
# ══════════════════════════════════════════════════════════════════════════════
class PLMDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray,
                 tokenizer, max_len: int, seq_formatter=fmt_standard):
        self.labels = labels
        formatted   = [seq_formatter(s) for s in sequences]
        self.encodings = tokenizer(
            formatted,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLM Classifier
# ══════════════════════════════════════════════════════════════════════════════
class PLMClassifier(nn.Module):
    def __init__(self, plm: nn.Module, hidden_size: int,
                 pooling: str = "cls", freeze_plm: bool = True):
        super().__init__()
        self.plm     = plm
        self.pooling = pooling

        if freeze_plm:
            for p in self.plm.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def _pool(self, last_hidden: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, **kwargs):
        accepted = self.plm.forward.__code__.co_varnames
        extra = {k: v for k, v in kwargs.items() if k in accepted}
        out = self.plm(input_ids=input_ids,
                       attention_mask=attention_mask, **extra)
        return self.head(self._pool(out.last_hidden_state, attention_mask))


# ══════════════════════════════════════════════════════════════════════════════
# 5. HuggingFace Model Loader
# ══════════════════════════════════════════════════════════════════════════════
def _hidden_size(model) -> int:
    cfg = model.config
    return getattr(cfg, "hidden_size", getattr(cfg, "d_model", getattr(cfg, "n_embd", 768)))

def load_hf_model(name: str) -> tuple:
    hf_id, family, max_len = HF_IDS[name]
    fmt    = SEQ_FORMATTERS[family]
    pool   = "cls" if family in ("bert", "esm") else "mean"

    log.info(f"  Loading {name} from '{hf_id}' ...")
    if family == "bert":
        tok = BertTokenizer.from_pretrained(hf_id, do_lower_case=False)
        plm = BertModel.from_pretrained(hf_id)
    elif family == "t5":
        tok = T5Tokenizer.from_pretrained(hf_id, do_lower_case=False)
        plm = T5EncoderModel.from_pretrained(hf_id)
    elif family == "esm":
        tok = EsmTokenizer.from_pretrained(hf_id)
        plm = EsmModel.from_pretrained(hf_id)
    else:
        tok = AutoTokenizer.from_pretrained(hf_id)
        plm = AutoModel.from_pretrained(hf_id)

    hidden = _hidden_size(plm)
    clf    = PLMClassifier(plm, hidden, pooling=pool, freeze_plm=FREEZE_PLM)
    return clf, tok, max_len, fmt


# ══════════════════════════════════════════════════════════════════════════════
# 6. Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def _make_loader(dataset: PLMDataset, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0, pin_memory=False)

def train_one_model(clf: PLMClassifier, name: str, fold: int,
                    train_ds: PLMDataset, val_ds: PLMDataset,
                    device: torch.device) -> dict:
    clf = clf.to(device)
    plm_params  = [p for p in clf.plm.parameters()  if p.requires_grad]
    head_params = list(clf.head.parameters())
    param_groups = [{"params": head_params, "lr": LR_HEAD}]
    if plm_params:
        param_groups.append({"params": plm_params, "lr": LR_PLM})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    best_val_acc, patience_ctr, best_state = -1.0, 0, None

    train_loader = _make_loader(train_ds, shuffle=True)
    val_loader   = _make_loader(val_ds,   shuffle=False)

    for epoch in range(EPOCHS):
        clf.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labs  = batch["label"].to(device)
            extra = {k: v.to(device) for k, v in batch.items() if k not in ("input_ids", "attention_mask", "label")}
            optimizer.zero_grad()
            out  = clf(ids, mask, **extra)
            loss = criterion(out, labs)
            loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item(); t_corr += out.argmax(1).eq(labs).sum().item()
            t_tot  += labs.size(0)

        t_loss /= len(train_loader); t_acc = t_corr / t_tot

        clf.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labs = batch["label"].to(device)
                extra = {k: v.to(device) for k, v in batch.items() if k not in ("input_ids", "attention_mask", "label")}
                out   = clf(ids, mask, **extra)
                v_loss += criterion(out, labs).item()
                v_corr += out.argmax(1).eq(labs).sum().item()
                v_tot  += labs.size(0)

        v_loss /= len(val_loader); v_acc = v_corr / v_tot
        scheduler.step(v_acc)

        history["train_loss"].append(t_loss); history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss);   history["val_acc"].append(v_acc)

        if (epoch + 1) % 5 == 0:
            log.info(f"    Fold {fold} - Epoch {epoch+1:>3}/{EPOCHS} | "
                     f"train loss={t_loss:.4f} acc={t_acc:.4f} | "
                     f"val loss={v_loss:.4f} acc={v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc; patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"    Fold {fold} - Early stopping at epoch {epoch+1}")
                break

    clf.load_state_dict(best_state)
    return history

def _infer(clf: PLMClassifier, dataset: PLMDataset,
           device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    clf.eval()
    loader = _make_loader(dataset, shuffle=False)
    preds, probas = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            extra = {k: v.to(device) for k, v in batch.items() if k not in ("input_ids", "attention_mask", "label")}
            out   = clf(ids, mask, **extra)
            prob  = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred  = out.argmax(1).cpu().numpy()
            preds.extend(pred); probas.extend(prob)
    return np.array(preds), np.array(probas)

def compute_metrics(y_true, y_pred, y_proba, fold_or_split: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prob2d = np.column_stack([1 - y_proba, y_proba])
    return {
        "fold":        fold_or_split,
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "mcc":         matthews_corrcoef(y_true, y_pred),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "log_loss":    log_loss(y_true, prob2d),
        "cm":          cm,
        "y_pred":      y_pred,
        "y_proba":     y_proba,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. ProteinBERT CV Implementation
# ══════════════════════════════════════════════════════════════════════════════
PROTEINBERT_SEQ_LEN = 512

class ProteinBERTCV:
    def __init__(self):
        if not PROTEINBERT_AVAILABLE:
            raise RuntimeError("ProteinBERT not installed. Run: pip install proteinbert")
        log.info("  Loading ProteinBERT pretrained weights ...")
        gen, self.input_encoder = load_pretrained_model()
        self.tf_model = gen.create_model(PROTEINBERT_SEQ_LEN)
        log.info("  ProteinBERT loaded (TF model).")

    def extract_embeddings(self, sequences: np.ndarray, batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = list(sequences[i: i + batch_size])
            enc_X, _ = self.input_encoder.encode_X(batch, PROTEINBERT_SEQ_LEN)
            annot    = np.zeros((len(batch), 8943), dtype=np.float32)
            outputs  = self.tf_model.predict([enc_X, annot], verbose=0)
            embeddings.append(outputs[1]) # [1] is the global vector
        return np.vstack(embeddings)

    def run_cv(self, X: np.ndarray, y: np.ndarray, skf, device: torch.device):
        log.info("  Extracting ProteinBERT embeddings for entire dataset...")
        all_embeddings = self.extract_embeddings(X)
        embed_dim = all_embeddings.shape[1]
        log.info(f"  ProteinBERT dynamic embedding dim inferred as: {embed_dim}")

        oof_pred = np.zeros(len(X))
        oof_proba = np.zeros(len(X))
        all_metrics_rows = []
        all_history_rows = []

        def to_ds(e, lbls):
            return TensorDataset(torch.tensor(e, dtype=torch.float32), torch.tensor(lbls, dtype=torch.long))

        loss_fn = nn.CrossEntropyLoss()

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y), 1):
            log.info(f"  --- Fold {fold}/{N_SPLITS} ---")
            
            # Split train_val into sub-train and sub-val (approx 10% for val)
            tr_idx, va_idx = train_test_split(train_val_idx, test_size=0.1111, random_state=RANDOM_STATE, stratify=y[train_val_idx])

            train_ld = DataLoader(to_ds(all_embeddings[tr_idx], y[tr_idx]), batch_size=64, shuffle=True)
            val_ld   = DataLoader(to_ds(all_embeddings[va_idx], y[va_idx]), batch_size=64)
            test_ld  = DataLoader(to_ds(all_embeddings[test_idx], y[test_idx]), batch_size=64)

            # Re-initialize head explicitly inside the fold to guarantee zero data leakage
            head = nn.Sequential(
                nn.Linear(embed_dim, 256), nn.LayerNorm(256),
                nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 2)
            ).to(device)

            opt = optim.AdamW(head.parameters(), lr=LR_HEAD, weight_decay=1e-4)

            history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
            best_acc, patience_ctr, best_state = -1.0, 0, None

            for epoch in range(EPOCHS):
                head.train()
                tl, tc, tt = 0.0, 0, 0
                for xb, yb in train_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    out  = head(xb); loss = loss_fn(out, yb); loss.backward()
                    opt.step()
                    tl += loss.item(); tc += out.argmax(1).eq(yb).sum().item(); tt += yb.size(0)
                ta = tc / tt; tl /= len(train_ld)

                head.eval(); vl, vc, vt = 0.0, 0, 0
                with torch.no_grad():
                    for xb, yb in val_ld:
                        xb, yb = xb.to(device), yb.to(device)
                        out = head(xb)
                        vl += loss_fn(out, yb).item()
                        vc += out.argmax(1).eq(yb).sum().item(); vt += yb.size(0)
                va = vc / vt; vl /= len(val_ld)

                history["train_loss"].append(tl); history["train_acc"].append(ta)
                history["val_loss"].append(vl);   history["val_acc"].append(va)

                if (epoch + 1) % 5 == 0:
                    log.info(f"    Fold {fold} - Epoch {epoch+1:>3}/{EPOCHS} | train loss={tl:.4f} acc={ta:.4f} | val loss={vl:.4f} acc={va:.4f}")

                if va > best_acc:
                    best_acc = va; patience_ctr = 0
                    best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                else:
                    patience_ctr += 1
                    if patience_ctr >= PATIENCE:
                        log.info(f"    Fold {fold} - Early stopping at epoch {epoch+1}")
                        break

            head.load_state_dict(best_state)
            head.eval()

            # Infer on Test
            preds, probs = [], []
            with torch.no_grad():
                for xb, _ in test_ld:
                    xb = xb.to(device)
                    out = head(xb)
                    probs.extend(F.softmax(out, 1)[:, 1].cpu().numpy())
                    preds.extend(out.argmax(1).cpu().numpy())
            
            # --- FIX: Convert to numpy arrays before math operations ---
            preds = np.array(preds)
            probs = np.array(probs)
            # -----------------------------------------------------------
            
            oof_pred[test_idx] = preds
            oof_proba[test_idx] = probs

            fold_m = compute_metrics(y[test_idx], preds, probs, f"fold_{fold}")
            fold_m["model"] = "ProteinBERT"
            all_metrics_rows.append(fold_m)

            for ep_i, (tloss, tacc, vloss, vacc) in enumerate(zip(history["train_loss"], history["train_acc"], history["val_loss"], history["val_acc"])):
                all_history_rows.append({
                    "model": "ProteinBERT", "fold": fold, "epoch": ep_i + 1,
                    "train_loss": tloss, "train_acc": tacc,
                    "val_loss": vloss,   "val_acc":   vacc,
                })

        return all_metrics_rows, all_history_rows, {"y_pred": oof_pred, "y_proba": oof_proba}


# ══════════════════════════════════════════════════════════════════════════════
# 8. Plots  
# ══════════════════════════════════════════════════════════════════════════════
def _model_dir(name: str, subfolder: str = "") -> Path:
    d = PLOTS_DIR / name.replace(" ", "_").replace("/", "_")
    if subfolder:
        d = d / subfolder
    d.mkdir(parents=True, exist_ok=True)
    return d

def plot_roc(name: str, y_test, y_proba, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(out_dir / f"{name.replace(' ', '_')}_roc.png", dpi=150); plt.close(fig)

def plot_confusion_matrix(cm: np.ndarray, name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    classes = ['Non-Antigen', 'Antigen']
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13, color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / f"{name.replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close(fig)

def plot_all_roc(all_predictions: dict, y_test: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_predictions)))
    for (name, preds), col in zip(all_predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
        ax.plot(fpr, tpr, lw=2, color=col, label=f'{name}  AUC={auc(fpr, tpr):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('Cross-Validation Aggregate ROC (OOF) — All PLMs', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "plm_cv_roc_all.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    log.info(f"  Combined CV ROC saved -> {path}")

def plot_performance_comparison(test_results: dict, out_dir: Path):
    metric_keys = ['accuracy', 'precision', 'specificity', 'recall', 'f1', 'mcc', 'roc_auc']
    models = list(test_results.keys())
    x      = np.arange(len(models))
    width  = 0.10
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_keys)))

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (mk, col) in enumerate(zip(metric_keys, colors)):
        vals = [test_results[m][mk] for m in models]
        ax.bar(x + i * width, vals, width, label=mk.replace('_', ' ').title(), color=col)

    cx = x + width * (len(metric_keys) - 1) / 2
    ax.set_xticks(cx)
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('CV OOF Performance Metrics — Protein Language Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = out_dir / "plm_cv_performance_comparison.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    log.info(f"  CV Performance comparison saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. PDF Report
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(summary_df: pd.DataFrame, out_dir: Path, plot_dir: Path):
    pdf_path = out_dir / "PLM_CV_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.75, 'Protein Language Models\nfor Antigen Prediction (10-Fold CV)', ha='center', fontsize=22, fontweight='bold')
        fig.text(0.5, 0.62, 'Plasmodium falciparum — PlasmoFAB Dataset', ha='center', fontsize=15)
        mode = "Feature Extraction" if FREEZE_PLM else "Full Fine-Tuning"
        fig.text(0.5, 0.55, f'Training strategy: {mode}', ha='center', fontsize=12, color='#555')
        fig.text(0.5, 0.48, f'Date: {datetime.now():%B %d, %Y}', ha='center', fontsize=12)
        models_list = '\n'.join(f'• {m}' for m in summary_df.index)
        fig.text(0.5, 0.20, 'Models Evaluated (OOF Test Metrics):\n' + models_list, ha='center', fontsize=10)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        disp = summary_df.round(4)
        tbl = ax.table(cellText=disp.values, colLabels=disp.columns, rowLabels=disp.index, cellLoc='center', loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 2)
        for j in range(len(disp.columns)):
            tbl[(0, j)].set_facecolor('#1A5276')
            tbl[(0, j)].set_text_props(weight='bold', color='white')
        ax.set_title('Cross-Validation Aggregate (OOF) Performance', fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        for pname in ['plm_cv_roc_all', 'plm_cv_performance_comparison']:
            path = plot_dir / f"{pname}.png"
            if path.exists():
                from PIL import Image
                img = Image.open(path)
                fig = plt.figure(figsize=(11, 8.5))
                plt.imshow(img); plt.axis('off')
                pdf.savefig(fig, bbox_inches='tight'); plt.close()

    log.info(f"  PDF report saved -> {pdf_path}")
    return pdf_path


# ══════════════════════════════════════════════════════════════════════════════
# 10. Main Execution Loop
# ══════════════════════════════════════════════════════════════════════════════
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    all_result_rows = []
    all_history_rows = []
    all_predictions  = {}     
    test_results_map = {}   

    # --- RESUME LOGIC ---
    metrics_file = RESULTS_DIR / "plm_cv_metrics.csv"
    hist_file = RESULTS_DIR / "plm_cv_training_history.csv"
    completed_models = []

    if metrics_file.exists():
        df_metrics = pd.read_csv(metrics_file)
        all_result_rows = df_metrics.to_dict('records')
        # If a model has an OOF_Aggregate row, it successfully finished all folds
        completed_models = df_metrics[df_metrics['fold'] == 'OOF_Aggregate']['model'].unique().tolist()
        
        # Rebuild the test map so the final Bar Chart has all 9 models
        for row in all_result_rows:
            if row['fold'] == 'OOF_Aggregate':
                test_results_map[row['model']] = row

    if hist_file.exists():
        all_history_rows = pd.read_csv(hist_file).to_dict('records')

    models_to_process = [m for m in MODELS_TO_RUN if m not in completed_models]

    log.info("=" * 65)
    log.info("PlasmoFAB — CV Protein Language Model Pipeline")
    log.info(f"  Already completed : {completed_models}")
    log.info(f"  Models running now: {models_to_process}")
    log.info(f"  CV Folds          : {N_SPLITS}")
    log.info("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Device: {device}")

    # ── Load Data & Setup KFold
    sequences, labels = load_data(INPUT_FILE)
    sequences, labels = preprocess(sequences, labels)
    X, y = np.array(sequences), np.array(labels)
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # ══════════════════════════════════════════════════════════════════════
    # ── HuggingFace models
    # ══════════════════════════════════════════════════════════════════════
    hf_models = [m for m in models_to_process if m in HF_IDS]
    if hf_models and not HF_AVAILABLE:
        log.error("transformers not installed — skipping HF models.")
        hf_models = []

    for name in hf_models:
        log.info(f"\n{'='*65}\n  Model: {name}\n{'='*65}")
        try:
            clf, tokenizer, max_len, fmt = load_hf_model(name)
            
            # Secure original untrained weights
            initial_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
            
            oof_pred = np.zeros(len(X))
            oof_proba = np.zeros(len(X))

            for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y), 1):
                log.info(f"\n  --- {name} | Fold {fold}/{N_SPLITS} ---")
                
                tr_idx, va_idx = train_test_split(train_val_idx, test_size=0.1111, random_state=RANDOM_STATE, stratify=y[train_val_idx])

                # Reset PyTorch Head Weights
                clf.load_state_dict(initial_state)

                train_ds = PLMDataset(X[tr_idx], y[tr_idx], tokenizer, max_len, fmt)
                val_ds   = PLMDataset(X[va_idx], y[va_idx], tokenizer, max_len, fmt)
                test_ds  = PLMDataset(X[test_idx], y[test_idx], tokenizer, max_len, fmt)

                history = train_one_model(clf, name, fold, train_ds, val_ds, device)
                torch.save(clf.state_dict(), CKPT_DIR / f"{name}_fold_{fold}_best.pth")

                te_pred, te_prob = _infer(clf, test_ds, device)
                oof_pred[test_idx] = te_pred
                oof_proba[test_idx] = te_prob

                # Document individual fold
                fold_m = compute_metrics(y[test_idx], te_pred, te_prob, f"fold_{fold}")
                fold_m["model"] = name
                all_result_rows.append(fold_m)

                for ep_i, (tl, ta, vl, va) in enumerate(zip(history["train_loss"], history["train_acc"], history["val_loss"],  history["val_acc"])):
                    all_history_rows.append({"model": name, "fold": fold, "epoch": ep_i + 1, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})

            # Evaluate OOF Aggregate
            oof_m = compute_metrics(y, oof_pred, oof_proba, "OOF_Aggregate")
            oof_m["model"] = name
            all_result_rows.append(oof_m)
            all_predictions[name] = {"y_pred": oof_pred, "y_proba": oof_proba}
            test_results_map[name] = oof_m

            # Plot OOF Curve for this model
            mdir_oof = _model_dir(name, "OOF_Aggregate")
            plot_roc(f"{name}_OOF", y, oof_proba, mdir_oof)
            plot_confusion_matrix(oof_m["cm"], f"{name}_OOF", mdir_oof)
            log.info(f"  OOF Plots saved -> {mdir_oof}/")

            # Cleanup
            del clf, train_ds, val_ds, test_ds
            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"  {name} FAILED: {e}", exc_info=True)
            continue

    # ══════════════════════════════════════════════════════════════════════
    # ── ProteinBERT
    # ══════════════════════════════════════════════════════════════════════
    if "ProteinBERT" in models_to_process:
        log.info(f"\n{'='*65}\n  Model: ProteinBERT (Brandes et al. 2022)\n{'='*65}")
        if not PROTEINBERT_AVAILABLE:
            log.warning("  Skipping — proteinbert not installed.")
        else:
            try:
                pb = ProteinBERTCV()
                metrics_rows, history_rows, oof_preds = pb.run_cv(X, y, skf, device)

                all_result_rows.extend(metrics_rows)
                all_history_rows.extend(history_rows)
                
                # Compute OOF for ProteinBERT
                oof_m = compute_metrics(y, oof_preds["y_pred"], oof_preds["y_proba"], "OOF_Aggregate")
                oof_m["model"] = "ProteinBERT"
                all_result_rows.append(oof_m)
                
                all_predictions["ProteinBERT"] = oof_preds
                test_results_map["ProteinBERT"] = oof_m

                mdir_oof = _model_dir("ProteinBERT", "OOF_Aggregate")
                plot_roc("ProteinBERT_OOF", y, oof_preds["y_proba"], mdir_oof)
                plot_confusion_matrix(oof_m["cm"], "ProteinBERT_OOF", mdir_oof)
                log.info(f"  ProteinBERT OOF Plots saved -> {mdir_oof}/")

            except Exception as e:
                log.error(f"  ProteinBERT FAILED: {e}", exc_info=True)

    # ══════════════════════════════════════════════════════════════════════
    # ── Summary Reports
    # ══════════════════════════════════════════════════════════════════════
    # Only try to replot the combined ROC if we generated multiple sets of probabilities in this specific run.
    # Otherwise we skip it to preserve the old 8-model ROC curve.
    if len(all_predictions) > 1:
        log.info("\nGenerating global CV summary plots ...")
        plot_all_roc(all_predictions, y, PLOTS_DIR)
        
    if test_results_map:
        plot_performance_comparison(test_results_map, PLOTS_DIR)

    metrics_df = pd.DataFrame(all_result_rows)
    metrics_df.to_csv(RESULTS_DIR / "plm_cv_metrics.csv", index=False)
    log.info(f"  Metrics saved -> {RESULTS_DIR / 'plm_cv_metrics.csv'}")

    pd.DataFrame(all_history_rows).to_csv(RESULTS_DIR / "plm_cv_training_history.csv", index=False)
    log.info(f"  Training history saved -> {RESULTS_DIR / 'plm_cv_training_history.csv'}")

    # Display Summary (Filtering ONLY for the OOF Aggregate rows)
    metric_cols = ['accuracy', 'precision', 'specificity', 'recall', 'f1', 'mcc', 'roc_auc', 'log_loss']
    test_summary = (
        metrics_df[metrics_df["fold"] == "OOF_Aggregate"]
        .set_index("model")
        .drop(columns="fold")[metric_cols]
        .sort_values("roc_auc", ascending=False)
    )

    log.info("\n" + "=" * 75)
    log.info("SUMMARY — CV OOF AGGREGATE")
    log.info("=" * 75)
    log.info("\n" + test_summary.round(4).to_string())
    log.info("=" * 75)

    if test_results_map:
        generate_pdf_report(test_summary, RESULTS_DIR, PLOTS_DIR)

    log.info("CV PLM pipeline complete.")

if __name__ == "__main__":
    main()
