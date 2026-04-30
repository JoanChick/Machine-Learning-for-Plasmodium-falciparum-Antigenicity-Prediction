#!/usr/bin/env python3

import sys
import os
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
PLOTS_DIR    = Path("plots/plm")
RESULTS_DIR  = Path("results")
CKPT_DIR     = Path("results/plm_checkpoints")

RANDOM_STATE = 0      # must match 00_preprocessing_feature_extraction.py
EPOCHS       = 50     # pretrained models converge faster than from-scratch
BATCH_SIZE   = 32     # reduce if OOM (try 8 or 4 for ProtT5/ESM-2-650M)
LR_HEAD      = 1e-3   # learning rate — classification head
LR_PLM       = 2e-5   # learning rate — PLM layers (used only if FREEZE_PLM=False)
PATIENCE     = 10      # early-stopping patience
FREEZE_PLM   = True   # True = feature extraction | False = full fine-tuning
MAX_LEN      = 1024    # default truncation; overridden per model where needed

# ── Models to run — comment out any to skip ───────────────────────────────────
MODELS_TO_RUN = [
    "ProtBERT",
    "ProtBERT-BFD",
    "ProtT5-XL",
    "ESM-1b",
    "ESM-2-8M",
    "ESM-2-35M",
    "ESM-2-650M",
    "Ankh-base",
    "ProteinBERT",    # requires TensorFlow + pip install proteinbert
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
# HF_IDS value: (hf_model_id, family, max_len)
# family controls tokenisation and pooling strategy


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"plm_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading  (identical logic to 00_preprocessing_feature_extraction.py)
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
    log.info(f"  Loaded {len(sequences)} sequences | "
             f"pos={sum(labels)}  neg={len(labels)-sum(labels)}")
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


def make_splits(sequences: list, labels: list) -> dict:
    """70 / 20 / 10 — same random_state as script 00."""
    X, y = np.array(sequences), np.array(labels)
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_STATE, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=0.2222, random_state=RANDOM_STATE, stratify=y_tmp)
    for split, (xs, ys) in [("train", (x_train, y_train)),
                             ("val",   (x_val,   y_val)),
                             ("test",  (x_test,  y_test))]:
        log.info(f"  {split:<6}: {len(xs):>4} | pos={ys.sum()}  neg={(ys==0).sum()}")
    return {"x_train": x_train, "y_train": y_train,
            "x_val":   x_val,   "y_val":   y_val,
            "x_test":  x_test,  "y_test":  y_test}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sequence Formatters
# ══════════════════════════════════════════════════════════════════════════════

def fmt_rostlab(seq: str) -> str:
    """
    ProtBERT / ProtT5 require space-separated AAs and rare-AA substitution.
    e.g. "MAGS" → "M A G S"
    """
    for rare in "UZOB":
        seq = seq.replace(rare, "X")
    return " ".join(seq)


def fmt_standard(seq: str) -> str:
    """ESM, Ankh — pass sequence as-is."""
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
    """
    Tokenises a list of protein sequences for any HuggingFace PLM.
    Tokenisation is done eagerly (at construction) to avoid per-epoch overhead.
    """

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
# 4. PLM Classifier (wraps any HuggingFace encoder)
# ══════════════════════════════════════════════════════════════════════════════

class PLMClassifier(nn.Module):
    """
    Wraps a pretrained HuggingFace encoder with a 2-layer classification head.

    Pooling strategies:
      'cls'  — use the [CLS] / <s> token representation  (BERT, ESM)
      'mean' — masked mean-pooling over all token positions (T5, Ankh)

    freeze_plm=True  → only the head is trained  (feature extraction)
    freeze_plm=False → all parameters trained    (full fine-tuning)
    """

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
        # masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, **kwargs):
        # Some models (BERT) accept token_type_ids; others don't.
        # We forward only what the model accepts.
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
    return getattr(cfg, "hidden_size",       # BERT / ESM
           getattr(cfg, "d_model",           # T5 family
           getattr(cfg, "n_embd", 768)))     # fallback


def load_hf_model(name: str) -> tuple:
    """
    Returns (PLMClassifier, tokenizer, max_len) for a named model.
    Raises RuntimeError if the model cannot be loaded.
    """
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

    n_total    = sum(p.numel() for p in clf.parameters())
    n_trainable= sum(p.numel() for p in clf.parameters() if p.requires_grad)
    log.info(f"  Parameters: {n_total:,}  |  Trainable: {n_trainable:,}  "
             f"|  Hidden: {hidden}  |  Pool: {pool}")

    return clf, tok, max_len, fmt


# ══════════════════════════════════════════════════════════════════════════════
# 6. Training & Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _make_loader(dataset: PLMDataset, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def train_one_model(clf: PLMClassifier, name: str,
                    train_ds: PLMDataset, val_ds: PLMDataset,
                    device: torch.device) -> dict:
    clf = clf.to(device)

    # Separate LRs for PLM and head (matters only when not frozen)
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
        # ── Train ────────────────────────────────────────────────────────────
        clf.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for batch in train_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labs  = batch["label"].to(device)
            extra = {k: v.to(device) for k, v in batch.items()
                     if k not in ("input_ids", "attention_mask", "label")}
            optimizer.zero_grad()
            out  = clf(ids, mask, **extra)
            loss = criterion(out, labs)
            loss.backward()
            nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item(); t_corr += out.argmax(1).eq(labs).sum().item()
            t_tot  += labs.size(0)

        t_loss /= len(train_loader); t_acc = t_corr / t_tot

        # ── Validate ─────────────────────────────────────────────────────────
        clf.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labs = batch["label"].to(device)
                extra = {k: v.to(device) for k, v in batch.items()
                         if k not in ("input_ids", "attention_mask", "label")}
                out   = clf(ids, mask, **extra)
                v_loss += criterion(out, labs).item()
                v_corr += out.argmax(1).eq(labs).sum().item()
                v_tot  += labs.size(0)

        v_loss /= len(val_loader); v_acc = v_corr / v_tot
        scheduler.step(v_acc)

        history["train_loss"].append(t_loss); history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss);   history["val_acc"].append(v_acc)

        if (epoch + 1) % 5 == 0:
            log.info(f"    Epoch {epoch+1:>3}/{EPOCHS} | "
                     f"train loss={t_loss:.4f} acc={t_acc:.4f} | "
                     f"val loss={v_loss:.4f} acc={v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc; patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"    Early stopping at epoch {epoch+1}")
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
            extra = {k: v.to(device) for k, v in batch.items()
                     if k not in ("input_ids", "attention_mask", "label")}
            out   = clf(ids, mask, **extra)
            prob  = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred  = out.argmax(1).cpu().numpy()
            preds.extend(pred); probas.extend(prob)
    return np.array(preds), np.array(probas)


def compute_metrics(y_true, y_pred, y_proba, split: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prob2d = np.column_stack([1 - y_proba, y_proba])
    return {
        "split":       split,
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
# 7. ProteinBERT (Brandes et al. 2022) — TensorFlow / Keras model
#    Feature-extraction only: embeddings → sklearn/PyTorch linear head
# ══════════════════════════════════════════════════════════════════════════════

PROTEINBERT_SEQ_LEN = 512   # includes 2 special tokens; model max = 2044

class ProteinBERTClassifier:
    """
    Wraps the Brandes et al. ProteinBERT (TF/Keras) as a feature extractor.
    Embeddings (1280-d global vectors) are extracted once, then a small
    PyTorch MLP head is trained on top.

    Architecture:
      Local (Conv, 512-d)  +  Global (Attention, 1280-d) layers
      We use the 1280-d *global* output → binary classifier head.
    """

    def __init__(self):
        if not PROTEINBERT_AVAILABLE:
            raise RuntimeError(
                "ProteinBERT not installed. Run: pip install proteinbert")

        log.info("  Loading ProteinBERT (Brandes et al.) pretrained weights ...")
        gen, self.input_encoder = load_pretrained_model()
        # Use create_model() directly — outputs exactly (local, global).
        # get_model_with_hidden_layers_as_outputs() concatenates ALL hidden
        # states into a single flat tensor, breaking the 1280-d shape assumption.
        self.tf_model = gen.create_model(PROTEINBERT_SEQ_LEN)
        log.info("  ProteinBERT loaded (TF model).")

    def extract_embeddings(self, sequences: np.ndarray,
                           batch_size: int = 32) -> np.ndarray:
        """
        Returns shape (N, global_dim) embeddings.
        create_model outputs (local, global); we use the global vector.
        global_dim = 1280 for the standard ProteinBERT release.
        """
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = list(sequences[i: i + batch_size])
            enc_X, _ = self.input_encoder.encode_X(batch, PROTEINBERT_SEQ_LEN)
            annot     = np.zeros((len(batch), 8943), dtype=np.float32)
            outputs   = self.tf_model.predict([enc_X, annot], verbose=0)
            # outputs is always a 2-element list: [local, global]
            # local  shape: (batch, seq_len, 512)
            # global shape: (batch, 1280)
            glob = outputs[1]
            embeddings.append(glob)
        return np.vstack(embeddings)

    def run(self, splits: dict) -> tuple[dict, dict, dict]:
        """
        Extract embeddings for all splits, train a PyTorch linear head,
        return (all_metrics_rows, train_metrics, val_metrics, test_metrics,
                history, test_predictions).
        """
        log.info("  Extracting ProteinBERT embeddings (train) ...")
        e_train = self.extract_embeddings(splits["x_train"])
        log.info("  Extracting ProteinBERT embeddings (val)   ...")
        e_val   = self.extract_embeddings(splits["x_val"])
        log.info("  Extracting ProteinBERT embeddings (test)  ...")
        e_test  = self.extract_embeddings(splits["x_test"])

        y_train = splits["y_train"]
        y_val   = splits["y_val"]
        y_test  = splits["y_test"]

        # Build MLP head — input dim inferred from actual embedding shape
        embed_dim = e_train.shape[1]
        log.info(f"  ProteinBERT embedding dim: {embed_dim}")
        head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.LayerNorm(256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        opt  = optim.AdamW(head.parameters(), lr=LR_HEAD, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        # Convert to tensors
        def to_ds(e, y):
            from torch.utils.data import TensorDataset
            return TensorDataset(
                torch.tensor(e, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long))

        train_ld = DataLoader(to_ds(e_train, y_train),
                              batch_size=64, shuffle=True)
        val_ld   = DataLoader(to_ds(e_val,   y_val),   batch_size=64)
        test_ld  = DataLoader(to_ds(e_test,  y_test),  batch_size=64)

        history  = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
        best_acc, best_state = -1.0, None

        for epoch in range(EPOCHS):
            head.train()
            tl, tc, tt = 0.0, 0, 0
            for xb, yb in train_ld:
                opt.zero_grad()
                out  = head(xb); loss = loss_fn(out, yb); loss.backward()
                opt.step()
                tl += loss.item(); tc += out.argmax(1).eq(yb).sum().item()
                tt += yb.size(0)
            ta = tc / tt; tl /= len(train_ld)

            head.eval(); vl, vc, vt = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_ld:
                    out = head(xb)
                    vl += loss_fn(out, yb).item()
                    vc += out.argmax(1).eq(yb).sum().item(); vt += yb.size(0)
            va = vc / vt; vl /= len(val_ld)

            history["train_loss"].append(tl); history["train_acc"].append(ta)
            history["val_loss"].append(vl);   history["val_acc"].append(va)

            if va > best_acc:
                best_acc = va
                best_state = {k: v.clone() for k, v in head.state_dict().items()}

        head.load_state_dict(best_state)
        head.eval()

        def infer_linear(loader):
            preds, probs = [], []
            with torch.no_grad():
                for xb, _ in loader:
                    out = head(xb)
                    probs.extend(F.softmax(out, 1)[:, 1].numpy())
                    preds.extend(out.argmax(1).numpy())
            return np.array(preds), np.array(probs)

        tr_pred, tr_prob = infer_linear(train_ld)
        va_pred, va_prob = infer_linear(val_ld)
        te_pred, te_prob = infer_linear(test_ld)

        metrics = {
            "train": compute_metrics(y_train, tr_pred, tr_prob, "train"),
            "val":   compute_metrics(y_val,   va_pred, va_prob, "validation"),
            "test":  compute_metrics(y_test,  te_pred, te_prob, "test"),
        }
        return metrics, history, {"y_pred": te_pred, "y_proba": te_prob}


# ══════════════════════════════════════════════════════════════════════════════
# 8. Plots  (matching style of 01_train_evaluate_models.py)
# ══════════════════════════════════════════════════════════════════════════════

def _model_dir(name: str) -> Path:
    d = PLOTS_DIR / name.replace(" ", "_").replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_roc(name: str, y_test, y_proba, out_dir: Path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'navy', lw=1.5, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve — {name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_roc.png", dpi=150); plt.close(fig)


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
    fig.savefig(out_dir / f"{name}_loss.png", dpi=150); plt.close(fig)


def plot_training_history(name: str, history: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (t_key, v_key), title in [
        (axes[0], ('train_loss', 'val_loss'), 'Loss'),
        (axes[1], ('train_acc',  'val_acc'),  'Accuracy'),
    ]:
        ax.plot(history[t_key], label='Train',      color='#4C72B0', lw=2)
        ax.plot(history[v_key], label='Validation', color='#DD8452', lw=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{name} — {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"{name}_training_history.png", dpi=150)
    plt.close(fig)


def plot_all_roc(all_predictions: dict, y_test: np.ndarray, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_predictions)))
    for (name, preds), col in zip(all_predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
        ax.plot(fpr, tpr, lw=2, color=col,
                label=f'{name}  AUC={auc(fpr, tpr):.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('ROC Curves — All Protein Language Models',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "plm_roc_all.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    log.info(f"  Combined ROC saved -> {path}")


def plot_performance_comparison(test_results: dict, out_dir: Path):
    metric_keys = ['accuracy', 'precision', 'specificity',
                   'recall', 'f1', 'mcc', 'roc_auc']
    models = list(test_results.keys())
    x      = np.arange(len(models))
    width  = 0.10
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_keys)))

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (mk, col) in enumerate(zip(metric_keys, colors)):
        vals = [test_results[m][mk] for m in models]
        ax.bar(x + i * width, vals, width,
               label=mk.replace('_', ' ').title(), color=col)

    cx = x + width * (len(metric_keys) - 1) / 2
    ax.set_xticks(cx)
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('Performance Metrics — Protein Language Models (Test Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = out_dir / "plm_performance_comparison.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    log.info(f"  Performance comparison saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. PDF Report
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(summary_df: pd.DataFrame,
                        out_dir: Path, plot_dir: Path):
    pdf_path = out_dir / "PLM_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.75,
                 'Protein Language Models\nfor Antigen Prediction',
                 ha='center', fontsize=22, fontweight='bold')
        fig.text(0.5, 0.62,
                 'Plasmodium falciparum — PlasmoFAB Dataset',
                 ha='center', fontsize=15)
        mode = "Feature Extraction" if FREEZE_PLM else "Full Fine-Tuning"
        fig.text(0.5, 0.55, f'Training strategy: {mode}',
                 ha='center', fontsize=12, color='#555')
        fig.text(0.5, 0.48, f'Date: {datetime.now():%B %d, %Y}',
                 ha='center', fontsize=12)
        models_list = '\n'.join(f'• {m}' for m in summary_df.index)
        fig.text(0.5, 0.20, 'Models Evaluated:\n' + models_list,
                 ha='center', fontsize=10)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # Results table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        disp = summary_df.round(4)
        tbl = ax.table(cellText=disp.values,
                       colLabels=disp.columns,
                       rowLabels=disp.index,
                       cellLoc='center', loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 2)
        for j in range(len(disp.columns)):
            tbl[(0, j)].set_facecolor('#1A5276')
            tbl[(0, j)].set_text_props(weight='bold', color='white')
        ax.set_title('Test Set Performance Summary',
                     fontsize=14, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

        # Embed saved plots
        for pname in ['plm_roc_all', 'plm_performance_comparison']:
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
# 10. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("PlasmoFAB — Protein Language Model Pipeline")
    log.info(f"  Training mode : {'Feature Extraction (PLM frozen)' if FREEZE_PLM else 'Full Fine-Tuning'}")
    log.info(f"  Models queued : {MODELS_TO_RUN}")
    log.info("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    sequences, labels = load_data(INPUT_FILE)
    sequences, labels = preprocess(sequences, labels)
    splits            = make_splits(sequences, labels)

    all_result_rows = []
    all_history_rows = []
    all_predictions  = {}     # name → {y_pred, y_proba}
    test_results_map = {}     # name → test metrics dict (for bar chart)

    # ══════════════════════════════════════════════════════════════════════
    # ── HuggingFace models ────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    hf_models = [m for m in MODELS_TO_RUN if m in HF_IDS]
    if hf_models and not HF_AVAILABLE:
        log.error("transformers not installed — skipping HF models.")
        hf_models = []

    for name in hf_models:
        log.info(f"\n{'='*65}")
        log.info(f"  Model: {name}")
        log.info(f"{'='*65}")
        try:
            clf, tokenizer, max_len, fmt = load_hf_model(name)

            train_ds = PLMDataset(splits["x_train"], splits["y_train"],
                                  tokenizer, max_len, fmt)
            val_ds   = PLMDataset(splits["x_val"],   splits["y_val"],
                                  tokenizer, max_len, fmt)
            test_ds  = PLMDataset(splits["x_test"],  splits["y_test"],
                                  tokenizer, max_len, fmt)

            history = train_one_model(clf, name, train_ds, val_ds, device)

            # Save best checkpoint
            torch.save(clf.state_dict(), CKPT_DIR / f"{name}_best.pth")

            # Evaluate all splits
            tr_pred, tr_prob = _infer(clf, train_ds, device)
            va_pred, va_prob = _infer(clf, val_ds,   device)
            te_pred, te_prob = _infer(clf, test_ds,  device)

            train_m = compute_metrics(splits["y_train"], tr_pred, tr_prob, "train")
            val_m   = compute_metrics(splits["y_val"],   va_pred, va_prob, "validation")
            test_m  = compute_metrics(splits["y_test"],  te_pred, te_prob, "test")

            # Log test metrics
            log.info(f"\n  ── Test results ──────────────────────────────")
            for k in ['accuracy', 'precision', 'specificity',
                      'recall', 'f1', 'mcc', 'roc_auc', 'log_loss']:
                log.info(f"  {k:<14}: {test_m[k]:.4f}")
            log.info(f"  Confusion Matrix:\n{test_m['cm']}")

            # Plots
            mdir = _model_dir(name)
            plot_roc(name, splits["y_test"], te_prob, mdir)
            plot_confusion_matrix(test_m["cm"], name, mdir)
            plot_loss_curve(name, train_m["log_loss"],
                            val_m["log_loss"], test_m["log_loss"], mdir)
            plot_training_history(name, history, mdir)
            log.info(f"  Plots saved -> {mdir}/")

            # Accumulate
            for m in [train_m, val_m, test_m]:
                all_result_rows.append({
                    "model":       name,
                    "split":       m["split"],
                    "accuracy":    m["accuracy"],
                    "precision":   m["precision"],
                    "specificity": m["specificity"],
                    "recall":      m["recall"],
                    "f1":          m["f1"],
                    "mcc":         m["mcc"],
                    "roc_auc":     m["roc_auc"],
                    "log_loss":    m["log_loss"],
                })

            all_predictions[name]  = {"y_pred": te_pred, "y_proba": te_prob}
            test_results_map[name] = test_m

            for ep_i, (tl, ta, vl, va) in enumerate(zip(
                    history["train_loss"], history["train_acc"],
                    history["val_loss"],  history["val_acc"])):
                all_history_rows.append({
                    "model": name, "epoch": ep_i + 1,
                    "train_loss": tl, "train_acc": ta,
                    "val_loss": vl,   "val_acc":   va,
                })

            # Free GPU memory between models
            del clf, train_ds, val_ds, test_ds
            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"  {name} FAILED: {e}", exc_info=True)
            continue

    # ══════════════════════════════════════════════════════════════════════
    # ── ProteinBERT (Brandes et al.) ──────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    if "ProteinBERT" in MODELS_TO_RUN:
        log.info(f"\n{'='*65}")
        log.info("  Model: ProteinBERT (Brandes et al. 2022)")
        log.info(f"{'='*65}")
        if not PROTEINBERT_AVAILABLE:
            log.warning("  Skipping — proteinbert not installed.\n"
                        "  Run: pip install proteinbert")
        else:
            try:
                pb = ProteinBERTClassifier()
                metrics_dict, history, test_preds = pb.run(splits)

                name = "ProteinBERT"
                mdir = _model_dir(name)

                # Log test metrics
                tm = metrics_dict["test"]
                log.info(f"\n  ── Test results ──────────────────────────────")
                for k in ['accuracy', 'precision', 'specificity',
                          'recall', 'f1', 'mcc', 'roc_auc', 'log_loss']:
                    log.info(f"  {k:<14}: {tm[k]:.4f}")

                # Plots
                plot_roc(name, splits["y_test"], test_preds["y_proba"], mdir)
                plot_confusion_matrix(tm["cm"], name, mdir)
                plot_loss_curve(
                    name,
                    metrics_dict["train"]["log_loss"],
                    metrics_dict["val"]["log_loss"],
                    tm["log_loss"], mdir)
                plot_training_history(name, history, mdir)
                log.info(f"  Plots saved -> {mdir}/")

                for split_key, m in metrics_dict.items():
                    all_result_rows.append({
                        "model":       name,
                        "split":       m["split"],
                        "accuracy":    m["accuracy"],
                        "precision":   m["precision"],
                        "specificity": m["specificity"],
                        "recall":      m["recall"],
                        "f1":          m["f1"],
                        "mcc":         m["mcc"],
                        "roc_auc":     m["roc_auc"],
                        "log_loss":    m["log_loss"],
                    })

                all_predictions[name]  = test_preds
                test_results_map[name] = tm

                for ep_i, (tl, ta, vl, va) in enumerate(zip(
                        history["train_loss"], history["train_acc"],
                        history["val_loss"],   history["val_acc"])):
                    all_history_rows.append({
                        "model": name, "epoch": ep_i + 1,
                        "train_loss": tl, "train_acc": ta,
                        "val_loss": vl,   "val_acc":   va,
                    })

            except Exception as e:
                log.error(f"  ProteinBERT FAILED: {e}", exc_info=True)

    # ══════════════════════════════════════════════════════════════════════
    # ── Summary plots + CSVs ──────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    if all_predictions:
        log.info("\nGenerating summary plots ...")
        plot_all_roc(all_predictions, splits["y_test"], PLOTS_DIR)
        plot_performance_comparison(test_results_map, PLOTS_DIR)

    # Save CSVs
    metrics_df = pd.DataFrame(all_result_rows)
    metrics_df.to_csv(RESULTS_DIR / "plm_metrics.csv", index=False)
    log.info(f"  Metrics saved -> {RESULTS_DIR / 'plm_metrics.csv'}")

    pd.DataFrame(all_history_rows).to_csv(
        RESULTS_DIR / "plm_training_history.csv", index=False)
    log.info(f"  Training history saved -> "
             f"{RESULTS_DIR / 'plm_training_history.csv'}")

    # Summary table
    metric_cols = ['accuracy', 'precision', 'specificity',
                   'recall', 'f1', 'mcc', 'roc_auc', 'log_loss']
    test_summary = (
        metrics_df[metrics_df["split"] == "test"]
        .set_index("model")
        .drop(columns="split")[metric_cols]
        .sort_values("roc_auc", ascending=False)
    )

    log.info("\n" + "=" * 75)
    log.info("SUMMARY — TEST SET")
    log.info("=" * 75)
    log.info("\n" + test_summary.round(4).to_string())
    log.info("=" * 75)

    if all_predictions:
        generate_pdf_report(test_summary, RESULTS_DIR, PLOTS_DIR)

    log.info("PLM pipeline complete.")


if __name__ == "__main__":
    main()
