#!/usr/bin/env python3

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from protlearn.features import (
    length, aac, aaindex1, ngram, entropy, atc, binary,
    posrich, cksaap, ctd, ctdc, ctdd, ctdt, moreau_broto,
    moran, geary, paac, apaac, socn, qso
)
from protlearn.preprocessing import remove_duplicates, remove_unnatural

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_FILE   = "PlasmoFAB_seq.csv"
OUTPUT_DIR   = Path("splits")
RANDOM_STATE = 0
LOG_LEVEL    = logging.INFO

FEATURE_GROUPS = {
    "length":         ["ln"],
    "composition":    ["comp", "aaind", "di", "tri", "ent", "atoms"],
    "binary":         ["bpp"],
    "position_rich":  ["pos_multiple"],
    "cksaap":         ["ck", "ck2"],
    "ctd":            ["ctd_arr", "c", "t", "d"],
    "autocorrelation":["mb", "moranI", "gearyC"],
    "pseudo_aac":     ["paac_comp", "apaac_comp"],
    "sequence_order": ["sw", "qw"],
}

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"feature_extraction_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────
def load_data(filepath: str) -> tuple[list, list]:
    """Load sequences and labels from PlasmoFAB CSV."""
    log.info(f"Loading data from '{filepath}' ...")
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    sequences, labels = [], []
    with open(filepath, 'r') as f:
        next(f)  # skip header
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


# ──────────────────────────────────────────────
# 2. Preprocessing
# ──────────────────────────────────────────────
def preprocess(sequences: list, labels: list) -> tuple[list, list]:
    """Remove duplicates and unnatural amino acids, keeping labels in sync."""
    log.info("Preprocessing sequences ...")
    n_before = len(sequences)

    # Remove duplicates manually to keep labels in sync
    seen = set()
    seqs_dedup, labels_dedup = [], []
    for seq, lbl in zip(sequences, labels):
        if seq not in seen:
            seen.add(seq)
            seqs_dedup.append(seq)
            labels_dedup.append(lbl)
    log.info(f"  Duplicates removed: {n_before - len(seqs_dedup)}")

    # Remove sequences with unnatural amino acids manually
    NATURAL = set("ACDEFGHIKLMNPQRSTVWY")
    seqs_clean, labels_clean = [], []
    for seq, lbl in zip(seqs_dedup, labels_dedup):
        if all(aa in NATURAL for aa in seq.strip().upper()):
            seqs_clean.append(seq)
            labels_clean.append(lbl)
    log.info(f"  Unnatural sequences removed: {len(seqs_dedup) - len(seqs_clean)}")
    log.info(f"  Sequences after preprocessing: {len(seqs_clean)}")

    return seqs_clean, labels_clean
def extract_features(seqs: list) -> tuple[np.ndarray, list]:
    """Extract and concatenate all feature groups. Returns array + column names."""
    log.info("Extracting features ...")
    arrays, col_names = [], []

    def add(arr, prefix):
        """Helper to collect array and generate column names."""
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
        col_names.extend([f"{prefix}_{i}" for i in range(arr.shape[1])])

    steps = [
        ("length",         lambda: (length(seqs),                              "ln")),
        ("AAC",            lambda: (aac(seqs, remove_zero_cols=True)[0],        "aac")),
        ("AAindex1",       lambda: (aaindex1(seqs, standardize='zscore')[0],    "aaind")),
        ("Digram",         lambda: (ngram(seqs, n=2)[0],                        "di")),
        ("Trigram",        lambda: (ngram(seqs, n=3)[0],                        "tri")),
        ("Entropy",        lambda: (entropy(seqs),                              "ent")),
        ("ATC",            lambda: (atc(seqs)[0],                               "atc")),
        ("Binary",         lambda: (binary(seqs, padding=True),                 "bin")),
        ("PosRich",        lambda: (posrich(seqs, position=[2,3,4], aminoacid=['R','N','L']), "posrich")),
        ("CKSAAP(k=1)",    lambda: (cksaap(seqs, remove_zero_cols=True)[0],     "ck1")),
        ("CKSAAP(k=2)",    lambda: (cksaap(seqs, k=2, remove_zero_cols=True)[0],"ck2")),
        ("CTD",            lambda: (ctd(seqs)[0],                               "ctd")),
        ("CTDC",           lambda: (ctdc(seqs)[0],                              "ctdc")),
        ("CTDT",           lambda: (ctdt(seqs)[0],                              "ctdt")),
        ("CTDD",           lambda: (ctdd(seqs)[0],                              "ctdd")),
        ("Moreau-Broto",   lambda: (moreau_broto(seqs),                         "mb")),
        ("Moran",          lambda: (moran(seqs),                                "moran")),
        ("Geary",          lambda: (geary(seqs),                                "geary")),
        ("PAAC",           lambda: (paac(seqs, lambda_=3, remove_zero_cols=True)[0],  "paac")),
        ("APAAC",          lambda: (apaac(seqs, lambda_=3, remove_zero_cols=True)[0], "apaac")),
        ("SOCN",           lambda: (socn(seqs, d=3)[0],                         "socn")),
        ("QSO",            lambda: (qso(seqs, d=3, remove_zero_cols=True)[0],   "qso")),
    ]

    for name, fn in steps:
        try:
            arr, prefix = fn()
            add(arr, prefix)
            log.info(f"  ✓ {name:<20} → {arr.shape[1] if arr.ndim > 1 else 1:>5} features")
        except Exception as e:
            log.error(f"  ✗ {name} FAILED: {e}")
            raise

    feature_matrix = np.concatenate(arrays, axis=1)
    log.info(f"  Total feature matrix shape: {feature_matrix.shape}")
    return feature_matrix, col_names


# ──────────────────────────────────────────────
# 4. Train / Validation / Test Split
# ──────────────────────────────────────────────
def split_and_save(features: np.ndarray, labels: list, col_names: list, output_dir: Path):
    """70 / 20 / 10 stratified split, saved as CSVs."""
    log.info("Splitting data (70% train | 20% val | 10% test) ...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Hold out 10% for test
    x_temp, x_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Step 2: Split remaining 90% into 70/20 (≈ 77.8/22.2 of the 90%)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        test_size=0.2222,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    splits = {
        "train":      (x_train, y_train),
        "validation": (x_val,   y_val),
        "test":       (x_test,  y_test),
    }

    total = len(features)
    for split_name, (X, y) in splits.items():
        df = pd.DataFrame(X, columns=col_names)
        df.insert(0, 'label', y)  # label as first column
        out_path = output_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)

        pos = sum(y); neg = len(y) - pos
        log.info(f"  {split_name:<12} → {len(X):>4} samples "
                 f"({len(X)/total*100:.1f}%) | "
                 f"pos={pos} neg={neg} | saved to '{out_path}'")

    return splits


# ──────────────────────────────────────────────
# 5. Summary
# ──────────────────────────────────────────────
def print_summary(features: np.ndarray, labels: list, splits: dict):
    log.info("=" * 55)
    log.info("SUMMARY")
    log.info(f"  Total sequences   : {len(features)}")
    log.info(f"  Total features    : {features.shape[1]}")
    log.info(f"  Class balance     : pos={sum(labels)} neg={len(labels)-sum(labels)}")
    for name, (X, y) in splits.items():
        log.info(f"  {name:<14}: {len(X)} samples")
    log.info("=" * 55)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 55)
    log.info("PlasmoFAB Feature Extraction Pipeline")
    log.info("=" * 55)

    sequences, labels     = load_data(INPUT_FILE)
    sequences, labels     = preprocess(sequences, labels)
    features, col_names   = extract_features(sequences)
    splits                = split_and_save(features, labels, col_names, OUTPUT_DIR)
    print_summary(features, labels, splits)

    log.info("Pipeline complete.")
