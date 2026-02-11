import torch, re, json, pandas as pd, numpy as np
from scipy.stats import rankdata
from transformers import AutoTokenizer, T5ForConditionalGeneration
from models.regressor import EncoderRegressor
from utils.data_loader import T5Dataset, EncDataset
import torch
print("CUDA inside job:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
import os
print("CONDA ENV:", os.environ.get("CONDA_DEFAULT_ENV"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_float(txt):
    m = re.search(r"\d+(\.\d+)?", txt)
    return float(m.group()) if m else 0.5

def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def run(split):
    df = pd.read_json(f"data/{split}.json").T.reset_index(drop=True)

    # T5
    t5_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
    t5.load_state_dict(torch.load("t5_model.pt"))
    t5.eval()

    t5_preds = []
    for b in T5Dataset(df, t5_tok):
        gen = t5.generate(b["input_ids"].unsqueeze(0).to(device))
        t5_preds.append(extract_float(t5_tok.decode(gen[0])) * 5)

    # Encoder
    enc_tok = AutoTokenizer.from_pretrained("roberta-base")
    enc = EncoderRegressor("roberta-base").to(device)
    enc.load_state_dict(torch.load("roberta_model.pt"))
    enc.eval()

    enc_preds = []
    for b in EncDataset(df, enc_tok):
        p = enc(b["input_ids"].unsqueeze(0).to(device), b["attention_mask"].unsqueeze(0).to(device))
        enc_preds.append(p.item())

    # Ensemble
    final = minmax(0.5 * np.sqrt(rankdata(t5_preds)) + 0.5 * rankdata(enc_preds)) * 5

    if split == "test":
        # Write predictions in required JSONL format for submission
        with open("predictions.jsonl", "w") as f:
            for i, v in enumerate(final):
                f.write(json.dumps({"id": str(i), "prediction": int(round(v))}) + "\n")
    else:
        np.save("dev_preds.npy", final)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
