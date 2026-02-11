import numpy as np
import torch
from torch.utils.data import Dataset

# ---------- helpers ----------
def safe_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)

def build_input(row):
    return (
        f"Sentence: {safe_str(row['sentence'])}\n"
        f"Meaning: {safe_str(row['judged_meaning'])}\n"
        f"Context: {safe_str(row['precontext'])} {safe_str(row['ending'])}"
    ).strip()

# ---------- T5 Dataset ----------
class T5Dataset(Dataset):
    def __init__(self, df, tokenizer):
        self.samples = []
        self.tok = tokenizer

        for _, row in df.iterrows():
            prompt = (
                "Task: Judge how natural and appropriate the meaning is "
                "for the sentence.\n"
                "Score from 0.0 to 1.0.\n"
                "ONLY output a number.\n\n"
                + build_input(row)
            )
            try:
                avg = float(row["average"])
            except Exception:
                avg = 0.5  # fallback default value
            label = round((avg / 5.0) * 20) / 20
            self.samples.append((prompt, f"{label:.2f}"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        enc_x = self.tok(x, truncation=True, max_length=256, return_tensors="pt")
        enc_y = self.tok(y, truncation=True, max_length=6, return_tensors="pt")
        return {
            "input_ids": enc_x["input_ids"].squeeze(0),
            "attention_mask": enc_x["attention_mask"].squeeze(0),
            "labels": enc_y["input_ids"].squeeze(0)
        }

# ---------- Encoder Dataset ----------
class EncDataset(Dataset):
    def __init__(self, df, tok):
        self.df = df
        self.tok = tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tok(
            build_input(row),
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        try:
            avg = float(row["average"])
        except Exception:
            avg = 0.5  # fallback default value
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(avg, dtype=torch.float)
        }
