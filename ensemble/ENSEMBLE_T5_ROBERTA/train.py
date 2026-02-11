import yaml, torch, pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorWithPadding
from models.regressor import EncoderRegressor
from models.losses import get_loss
from utils.data_loader import T5Dataset, EncDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
cfg = yaml.safe_load(open("configs/model_config.yaml"))
# ===== FIX CONFIG TYPES =====
cfg["training"]["t5_lr"] = float(cfg["training"]["t5_lr"])
cfg["training"]["encoder_lr"] = float(cfg["training"]["encoder_lr"])
cfg["training"]["batch_size"] = int(cfg["training"]["batch_size"])
cfg["training"]["encoder_epochs"] = int(cfg["training"]["encoder_epochs"])
cfg["training"]["t5_epochs"] = int(cfg["training"]["t5_epochs"])
print("AFTER CAST t5_lr:", cfg["training"]["t5_lr"], type(cfg["training"]["t5_lr"]))
print("AFTER CAST encoder_lr:", cfg["training"]["encoder_lr"], type(cfg["training"]["encoder_lr"]))


train_df = pd.read_json(cfg["paths"]["train_data"]).T.reset_index(drop=True)

# Data collators for padding
t5_data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(cfg["models"]["t5"]))

# Custom collate function for encoder
from torch.nn.utils.rnn import pad_sequence
def enc_collate_fn(batch):
    return {
        "input_ids": pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0),
        "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0),
        "label": torch.stack([b["label"] for b in batch])
    }


# ---- Train T5 ----
t5_tok = AutoTokenizer.from_pretrained(cfg["models"]["t5"])
from transformers import AutoConfig
config = AutoConfig.from_pretrained(cfg["models"]["t5"])
config.tie_word_embeddings = False
t5_model = T5ForConditionalGeneration.from_pretrained(cfg["models"]["t5"], config=config).to(device)

t5_dataset = T5Dataset(train_df, t5_tok)

t5_loader = DataLoader(
    t5_dataset,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True,
    collate_fn=t5_data_collator
)

opt_t5 = torch.optim.AdamW(t5_model.parameters(), lr=cfg["training"]["t5_lr"])

for _ in range(cfg["training"]["t5_epochs"]):
    for b in t5_loader:
        opt_t5.zero_grad()
        loss = t5_model(
            b["input_ids"].to(device),
            b["attention_mask"].to(device),
            labels=b["labels"].to(device)
        ).loss
        loss.backward()
        opt_t5.step()



# ---- Train Encoder ----
enc_tok = AutoTokenizer.from_pretrained(cfg["models"]["encoder"])
enc_model = EncoderRegressor(cfg["models"]["encoder"]).to(device)

enc_loader = DataLoader(
    EncDataset(train_df, enc_tok),
    batch_size=cfg["training"]["batch_size"],
    shuffle=True,
    collate_fn=enc_collate_fn
)
loss_fn = get_loss()
opt_enc = torch.optim.AdamW(enc_model.parameters(), lr=cfg["training"]["encoder_lr"])

for _ in range(cfg["training"]["encoder_epochs"]):
    for b in enc_loader:
        opt_enc.zero_grad()
        preds = enc_model(b["input_ids"].to(device), b["attention_mask"].to(device))
        loss = loss_fn(preds, b["label"].to(device))
        loss.backward()
        opt_enc.step()


# ===== SAVE TRAINED MODELS =====

# ===== SAVE TRAINED MODELS =====

torch.save(t5_model.state_dict(), "t5_model.pt")
torch.save(enc_model.state_dict(), "roberta_model.pt")

print("✅ Models saved:")
print(" - t5_model.pt")
print(" - roberta_model.pt")
