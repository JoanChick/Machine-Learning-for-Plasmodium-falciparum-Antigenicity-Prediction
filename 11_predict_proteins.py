#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
# Using the specific "best" checkpoint you mentioned earlier
FASTA_PATH = "/plasmodium_3d7_proteome.fasta"
MODEL_PATH = "plm_checkpoints/ESM-2-650M_best.pth"
HF_ID      = "facebook/esm2_t33_650M_UR50D"
OUTPUT_CSV = "plasmodium_3d7_single_model_predictions.csv"
MAX_LEN    = 1024
BATCH_SIZE = 1  # Single-sequence processing for CPU stability

# Explicitly set to CPU for your current environment
device = torch.device('cpu')
torch.set_grad_enabled(False)

# ──────────────────────────────────────────────
# 1. Model Architecture (Must match your 06_plm_cv.py)
# ──────────────────────────────────────────────
class PLMClassifier(nn.Module):
    def __init__(self, plm, hidden_size):
        super().__init__()
        self.plm = plm
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

    def forward(self, input_ids, attention_mask):
        out = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        # Use mean pooling as defined in your PLMClassifier class
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)

# ──────────────────────────────────────────────
# 2. Main Inference Loop
# ──────────────────────────────────────────────
def main():
    print(f"Loading Tokenizer and ESM-2-650M Base Model...")
    tokenizer = EsmTokenizer.from_pretrained(HF_ID)
    base_model = EsmModel.from_pretrained(HF_ID)
    model = PLMClassifier(base_model, base_model.config.hidden_size).to(device)
    
    print(f"Loading weights from: {MODEL_PATH}")
    # map_location='cpu' is vital here since no GPU is available
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load FASTA
    records = list(SeqIO.parse(FASTA_PATH, "fasta"))
    print(f"Total sequences to classify: {len(records)}")

    results = []

    # Inference loop
    for record in tqdm(records, desc="Predicting Antigens"):
        seq = str(record.seq).upper()
        
        # Tokenize single sequence
        inputs = tokenizer(
            [seq], 
            padding=True, 
            truncation=True, 
            max_length=MAX_LEN, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            # Antigen is Class 1
            prob = F.softmax(logits, dim=1)[0, 1].item()
            pred = logits.argmax(dim=1).item()

        results.append({
            "protein_id": record.id,
            "antigen_probability": round(prob, 4),
            "prediction": "Antigen" if pred == 1 else "Non-Antigen"
        })
        
        # Periodic memory cleanup for long FASTA files
        if len(results) % 100 == 0:
            gc.collect()

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Results saved to: {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
