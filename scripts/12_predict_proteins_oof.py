#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import gc

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
FASTA_PATH = "/home/bioinformatics/t/antigen_prediction_q1/update/update/plasmodium_3d7_proteome.fasta"
CKPT_PATTERN = "results/plm_cv_20260421_175917/checkpoints/ESM-2-650M_fold_*_best.pth"
HF_ID      = "facebook/esm2_t33_650M_UR50D"
OUTPUT_CSV = "plasmodium_3d7_oof_ensemble_gpu.csv"
MAX_LEN    = 1024
BATCH_SIZE = 16  # Increased for A100 GPU efficiency

device = torch.device('cuda')

# 1. Model Architecture
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
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)

def main():
    tokenizer = EsmTokenizer.from_pretrained(HF_ID)
    base_model = EsmModel.from_pretrained(HF_ID)
    model = PLMClassifier(base_model, base_model.config.hidden_size).to(device)
    
    ckpt_files = sorted(glob.glob(CKPT_PATTERN))
    records = list(SeqIO.parse(FASTA_PATH, "fasta"))
    all_fold_probs = np.zeros((len(records), len(ckpt_files)))

    # Process each fold
    for fold_idx, ckpt_path in enumerate(ckpt_files):
        print(f"\nProcessing Fold {fold_idx + 1}/10: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(records), BATCH_SIZE)):
                batch = records[i : i + BATCH_SIZE]
                seqs = [str(r.seq).upper() for r in batch]
                
                inputs = tokenizer(seqs, padding=True, truncation=True, 
                                   max_length=MAX_LEN, return_tensors="pt").to(device)
                
                logits = model(inputs['input_ids'], inputs['attention_mask'])
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_fold_probs[i : i + len(batch), fold_idx] = probs
        
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate Aggregate OOF Probability
    mean_probs = all_fold_probs.mean(axis=1)
    
    # Save Results
    results = []
    for idx, r in enumerate(records):
        results.append({
            "protein_id": r.id,
            "antigen_probability": round(float(mean_probs[idx]), 4),
            "prediction": "Antigen" if mean_probs[idx] >= 0.5 else "Non-Antigen"
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nGPU OOF Ensemble complete! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
