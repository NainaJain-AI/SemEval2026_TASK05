import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

gold = pd.read_json("data/dev.json").T["average"].astype(float).to_numpy()
preds = np.load("dev_preds.npy")

print("Pearson :", pearsonr(preds, gold)[0])
print("Spearman:", spearmanr(preds, gold)[0])
