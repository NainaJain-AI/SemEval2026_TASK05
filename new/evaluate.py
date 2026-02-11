import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    gold = pd.read_json("data/dev.json").T["average"].astype(float).to_numpy()
    preds = np.load("dev_preds.npy")
    
    pearson_r, pearson_p = pearsonr(preds, gold)
    spearman_r, spearman_p = spearmanr(preds, gold)
    
    print("Evaluation Results:")
    print("=" * 40)
    print(f"Pearson  r: {pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4e})")
    print("=" * 40)
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure to run prediction on dev split first:")
    print("  python predict.py dev")

