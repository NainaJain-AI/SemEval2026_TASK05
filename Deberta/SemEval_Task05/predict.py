"""
Prediction/inference script for T5-v1 AmbiStory model.

Usage:
    python predict.py --config configs/model_config.yaml --test_path test.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from models import ImprovedSemEvalModel
from utils import SemEvalDataset
from utils.data_loader import get_special_tokens, LABEL_MEAN, LABEL_STD


class EnsembleModel:
    """Ensemble predictions from multiple fold models."""
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        """
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal)
        """
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
    
    def predict(self, loader, device) -> np.ndarray:
        """Generate ensemble predictions by averaging."""
        all_preds = []
        
        for model_idx, model in enumerate(self.models):
            model.eval()
            fold_preds = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Model {model_idx+1}/{len(self.models)}"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    fold_preds.extend(outputs.cpu().numpy())
            
            all_preds.append(np.array(fold_preds))
        
        # Weighted average
        ensemble_preds = np.zeros_like(all_preds[0])
        for pred, weight in zip(all_preds, self.weights):
            ensemble_preds += pred * weight
        
        # Denormalize
        ensemble_preds = ensemble_preds * LABEL_STD + LABEL_MEAN
        ensemble_preds = np.clip(ensemble_preds, 1.0, 5.0)
        
        return ensemble_preds


def load_models(model_dir: Path, config: dict, device, num_folds: int = 5):
    """Load all fold models."""
    models = []
    special_tokens = get_special_tokens()
    
    for fold in range(1, num_folds + 1):
        model_path = model_dir / f"model_fold_{fold}.pt"
        if not model_path.exists():
            print(f"Warning: Model not found: {model_path}")
            continue
        
        model = ImprovedSemEvalModel(
            config['model']['name'],
            num_added_tokens=len(special_tokens['additional_special_tokens']),
            dropout=config['model']['dropout']
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded: {model_path}")
    
    return models


def main():
    parser = argparse.ArgumentParser(description="T5-v1 AmbiStory Prediction")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to saved models (default: from config)"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for predictions (default: from config)"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds to ensemble"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Paths
    model_dir = Path(args.model_dir or config['output']['model_dir'])
    output_file = args.output_file or config['output']['predictions_file']
    
    # Device
    device = torch.device(
        config['hardware']['device'] 
        if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer_path = model_dir / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        special_tokens = get_special_tokens()
        tokenizer.add_special_tokens(special_tokens)
    
    # Load test data
    print(f"\nLoading test data from: {args.test_path}")
    test_df = pd.read_json(args.test_path).T.reset_index(drop=True)
    print(f"Test samples: {len(test_df)}")
    
    # Create dataset
    test_dataset = SemEvalDataset(
        test_df, tokenizer,
        max_len=config['model']['max_len'],
        augmenter=None,
        is_training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'] and device.type == 'cuda',
        persistent_workers=config['hardware']['num_workers'] > 0
    )
    
    # Load models
    print(f"\nLoading models from: {model_dir}")
    models = load_models(model_dir, config, device, args.n_folds)
    
    if not models:
        print("Error: No models found!")
        return
    
    print(f"Loaded {len(models)} models for ensemble")
    
    # Create ensemble and predict
    ensemble = EnsembleModel(models)
    
    print("\nGenerating predictions...")
    predictions = ensemble.predict(test_loader, device)
    
    # Create submission format
    submission = {}
    for idx, (_, row) in enumerate(test_df.iterrows()):
        sample_id = str(row.get('sample_id', idx))
        submission[sample_id] = float(predictions[idx])
    
    # Save predictions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"  Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"  Prediction mean: {predictions.mean():.2f}")
    print(f"  Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
