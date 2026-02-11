import torch
import re
import json
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_loader import LlamaFewShotDataset, load_few_shot_examples

print("CUDA inside job:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

import os
print("CONDA ENV:", os.environ.get("CONDA_DEFAULT_ENV"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_rating(txt):
    """Extract numerical rating from model output"""
    # Try to find any number from 1-5
    matches = re.findall(r'\b[1-5]\b', txt)
    if matches:
        return int(matches[0])
    
    # Fallback: try any digit
    match = re.search(r'\d+', txt)
    if match:
        rating = int(match.group())
        return max(1, min(5, rating))  # Clamp to 1-5
    
    return 3  # Default middle rating

def run(split):
    print(f"\nProcessing {split} split...")
    
    df = pd.read_json(f"data/{split}.json").T.reset_index(drop=True)
    print(f"Loaded {len(df)} samples from {split} split")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    ).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load("checkpoints/llama_model.pt", map_location=device))
        print("✅ Loaded trained model from checkpoints/llama_model.pt")
    except FileNotFoundError:
        print("⚠️  Model checkpoint not found. Using pre-trained model.")
    
    model.eval()
    
    # Load few-shot examples for in-context learning
    few_shot_examples = df.head(5).to_dict('records')
    
    predictions = []
    
    with torch.no_grad():
        for idx, (_, row) in enumerate(df.iterrows()):
            # Build prompt with few-shot examples
            prompt = (
                "Task: Rate how natural and appropriate the meaning is for the sentence.\n"
                "Provide a rating from 1 to 5.\n\n"
                "Examples:\n"
            )
            
            for i, ex in enumerate(few_shot_examples, 1):
                prompt += (
                    f"Example {i}:\n"
                    f"Sentence: {ex.get('sentence', '')}\n"
                    f"Meaning: {ex.get('judged_meaning', '')}\n"
                    f"Rating: {int(round(float(ex.get('average', 2.5))))}\n\n"
                )
            
            prompt += (
                f"Now rate this:\n"
                f"Sentence: {row.get('sentence', '')}\n"
                f"Meaning: {row.get('judged_meaning', '')}\n"
                f"Rating: "
            )
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            # Generate
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract rating
            rating = extract_rating(generated_text)
            predictions.append(rating)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples")
    
    predictions = np.array(predictions)
    
    if split == "test":
        # Write predictions in JSONL format for submission
        with open("predictions.jsonl", "w") as f:
            for i, pred in enumerate(predictions):
                f.write(json.dumps({"id": str(i), "prediction": int(pred)}) + "\n")
        print(f"✅ Predictions saved to predictions.jsonl")
    else:
        # Save for evaluation
        np.save(f"{split}_preds.npy", predictions)
        print(f"✅ Predictions saved to {split}_preds.npy")
    
    return predictions

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        split = sys.argv[1]
        run(split)
    else:
        print("Usage: python predict.py <split>")
        print("  split: 'dev' or 'test'")

