import numpy as np
import torch
from torch.utils.data import Dataset
import json

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

# ---------- Few-Shot Examples Loader ----------
def load_few_shot_examples(examples_path, num_shots=5):
    """Load few-shot examples from JSON file"""
    try:
        with open(examples_path, 'r') as f:
            examples = json.load(f)
        return examples[:num_shots]
    except FileNotFoundError:
        print(f"Warning: Few-shot examples not found at {examples_path}")
        return []

def build_few_shot_prompt(examples, num_shots=5):
    """Build few-shot prompt from examples"""
    prompt = "Here are some examples of rating word meanings:\n\n"
    
    for i, ex in enumerate(examples[:num_shots], 1):
        prompt += f"Example {i}:\n"
        prompt += build_input(ex) + "\n"
        try:
            rating = float(ex.get("average", 0.5))
        except:
            rating = 0.5
        prompt += f"Rating: {rating:.1f}/5\n\n"
    
    return prompt

# ---------- LLAMA Few-Shot Dataset ----------
class LlamaFewShotDataset(Dataset):
    def __init__(self, df, tokenizer, few_shot_examples=None, num_shots=5):
        self.samples = []
        self.tok = tokenizer
        self.few_shot_prompt = ""
        
        if few_shot_examples:
            self.few_shot_prompt = build_few_shot_prompt(few_shot_examples, num_shots)

        for _, row in df.iterrows():
            # Build task instruction with few-shot examples
            task_instruction = (
                "Task: Rate how natural and appropriate the meaning is for the sentence.\n"
                "Provide a rating from 1 to 5, where:\n"
                "1 = Not natural at all\n"
                "3 = Moderately natural\n"
                "5 = Very natural and appropriate\n\n"
            )
            
            prompt = task_instruction + self.few_shot_prompt + "Now rate this:\n" + build_input(row)
            
            try:
                avg = float(row["average"])
            except Exception:
                avg = 2.5  # fallback default value
            
            # Target rating
            target = f"{int(round(avg))}"  # Round to nearest integer 1-5
            
            self.samples.append((prompt, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        enc_x = self.tok(x, truncation=True, max_length=512, return_tensors="pt")
        enc_y = self.tok(y, truncation=True, max_length=10, return_tensors="pt")
        
        return {
            "input_ids": enc_x["input_ids"].squeeze(0),
            "attention_mask": enc_x["attention_mask"].squeeze(0),
            "labels": enc_y["input_ids"].squeeze(0)
        }

