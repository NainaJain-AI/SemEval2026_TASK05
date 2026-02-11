import yaml
import torch
import pandas as pd
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from utils.data_loader import LlamaFewShotDataset, load_few_shot_examples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

cfg = yaml.safe_load(open("configs/model_config.yaml"))

# Fix config types
cfg["training"]["llama_lr"] = float(cfg["training"]["llama_lr"])
cfg["training"]["batch_size"] = int(cfg["training"]["batch_size"])
cfg["training"]["epochs"] = int(cfg["training"]["epochs"])
cfg["training"]["num_few_shots"] = int(cfg["training"]["num_few_shots"])

print("Config loaded:")
print(f"  Model: {cfg['models']['llama']}")
print(f"  Learning rate: {cfg['training']['llama_lr']}")
print(f"  Batch size: {cfg['training']['batch_size']}")
print(f"  Epochs: {cfg['training']['epochs']}")
print(f"  Few-shot examples: {cfg['training']['num_few_shots']}")

# Load data
train_df = pd.read_json(cfg["paths"]["train_data"]).T.reset_index(drop=True)
print(f"Loaded {len(train_df)} training samples")

# Load few-shot examples from training data
few_shot_examples = train_df.head(cfg["training"]["num_few_shots"]).to_dict('records')

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(cfg["models"]["llama"])
tokenizer.pad_token = tokenizer.eos_token  # For causal LM

model = AutoModelForCausalLM.from_pretrained(
    cfg["models"]["llama"],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
).to(device)

# Create dataset
dataset = LlamaFewShotDataset(
    train_df,
    tokenizer,
    few_shot_examples=few_shot_examples,
    num_shots=cfg["training"]["num_few_shots"]
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True,
    collate_fn=data_collator
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["training"]["llama_lr"]
)

# Training loop
print("\nStarting training...")
model.train()

for epoch in range(cfg["training"]["epochs"]):
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
        optimizer.step()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{cfg['training']['epochs']}, "
                  f"Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}\n")

# Save model
print("✅ Saving model...")
torch.save(model.state_dict(), "checkpoints/llama_model.pt")
print("✅ Model saved to checkpoints/llama_model.pt")

