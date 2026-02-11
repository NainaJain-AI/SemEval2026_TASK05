"""
Main training script for T5-v1 AmbiStory model.

Usage:
    python train.py --config configs/model_config.yaml
    python train.py --config configs/model_config.yaml --dry_run
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models import ImprovedSemEvalModel, CombinedLoss
from utils import SemEvalDataset, DataAugmenter
from utils.data_loader import get_special_tokens, compute_label_statistics, LABEL_MEAN, LABEL_STD


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta
    
    def load_best_model(self, model):
        model.load_state_dict(self.best_model)


def evaluate(model, loader, device, label_mean=LABEL_MEAN, label_std=LABEL_STD):
    """Comprehensive evaluation with all task metrics."""
    model.eval()
    all_preds, all_golds, all_stds, all_raw = [], [], [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            all_preds.extend(outputs.cpu().numpy())
            all_golds.extend(batch["label"].numpy())
            all_stds.extend(batch["std"].numpy())
            all_raw.extend(batch["raw_label"].numpy())
    
    # Denormalize predictions
    preds = np.array(all_preds) * label_std + label_mean
    golds = np.array(all_raw)  # Use raw labels
    stds = np.array(all_stds)
    
    # Clip predictions to valid range
    preds = np.clip(preds, 1.0, 5.0)
    
    # Metrics
    pearson = pearsonr(preds, golds)[0]
    spearman = spearmanr(preds, golds)[0]
    
    # Accuracy within standard deviation (at least 1)
    correct = sum(1 for p, g, s in zip(preds, golds, stds) if abs(p - g) <= max(s, 1.0))
    acc_std = correct / len(preds)
    
    mae = np.mean(np.abs(preds - golds))
    rmse = np.sqrt(np.mean((preds - golds) ** 2))
    
    return {
        'pearson': pearson,
        'spearman': spearman,
        'acc_std': acc_std,
        'mae': mae,
        'rmse': rmse,
        'predictions': preds,
        'gold': golds
    }


def train_fold(model, train_loader, val_loader, config, device, fold_num=0):
    """Train a single fold with all improvements."""
    
    # Optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and not any(nd in n for nd in no_decay)],
            'weight_decay': config['training']['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=config['training']['learning_rate']
    )
    
    # Learning rate scheduler
    total_steps = (
        len(train_loader) * config['training']['epochs'] 
        // config['training']['accumulation_steps']
    )
    warmup_steps = int(total_steps * config['training']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = CombinedLoss(
        mse_weight=config['loss']['mse_weight'],
        ordinal_weight=config['loss']['ordinal_weight']
    ).to(device)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['training']['patience'], mode='max')
    
    # Mixed precision (PyTorch 2.0+ API)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    
    best_metrics = None
    history = []
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Fold {fold_num+1} Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            stds = batch["std"].to(device)
            raw_labels = batch["raw_label"].to(device)
            
            # Forward pass with mixed precision
            if use_amp:
                with torch.amp.autocast('cuda'):
                    preds = model(input_ids, attention_mask)
                    loss = criterion(preds, labels, stds, raw_labels)
                    loss = loss / config['training']['accumulation_steps']
                
                scaler.scale(loss).backward()
            else:
                preds = model(input_ids, attention_mask)
                loss = criterion(preds, labels, stds, raw_labels)
                loss = loss / config['training']['accumulation_steps']
                loss.backward()
            
            total_loss += loss.item() * config['training']['accumulation_steps']
            
            # Gradient accumulation
            if (step + 1) % config['training']['accumulation_steps'] == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluation
        metrics = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            **{k: v for k, v in metrics.items() if k not in ['predictions', 'gold']}
        })
        
        # Print progress
        print(
            f"Fold {fold_num+1} | Epoch {epoch+1}/{config['training']['epochs']} | "
            f"Loss: {avg_loss:.4f} | "
            f"Spearman: {metrics['spearman']:.4f} | "
            f"Pearson: {metrics['pearson']:.4f} | "
            f"Acc@Std: {metrics['acc_std']:.4f}"
        )
        
        # Early stopping check
        if early_stopping(metrics['spearman'], model):
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break
        
        if best_metrics is None or metrics['spearman'] > best_metrics['spearman']:
            best_metrics = {k: v for k, v in metrics.items() if k not in ['predictions', 'gold']}
    
    # Load best model
    early_stopping.load_best_model(model)
    
    return model, best_metrics, history


def train_with_kfold(full_df, config, tokenizer, augmenter, device, output_dir):
    """Train with K-Fold cross-validation."""
    
    n_folds = config['training']['n_folds']
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config['seed'])
    
    # fold_models = []  # Removed to save memory
    fold_metrics = []
    all_oof_preds = np.zeros(len(full_df))
    
    special_tokens = get_special_tokens()
    
    print("=" * 60)
    print(f"Starting {n_folds}-Fold Cross-Validation")
    print("=" * 60)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_df)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Split data
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        # Create datasets
        train_dataset = SemEvalDataset(
            train_df, tokenizer,
            max_len=config['model']['max_len'],
            augmenter=augmenter if config['augmentation']['enabled'] else None,
            is_training=True
        )
        val_dataset = SemEvalDataset(
            val_df, tokenizer,
            max_len=config['model']['max_len'],
            augmenter=None,
            is_training=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory'] and device.type == 'cuda',
            persistent_workers=config['hardware']['num_workers'] > 0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'] * 2,
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory'] and device.type == 'cuda',
            persistent_workers=config['hardware']['num_workers'] > 0
        )
        
        # Initialize model
        model = ImprovedSemEvalModel(
            config['model']['name'],
            num_added_tokens=len(special_tokens['additional_special_tokens']),
            dropout=config['model']['dropout']
        ).to(device)
        
        # Freeze encoder layers
        model.freeze_encoder(num_unfrozen_layers=config['training']['num_unfrozen_layers'])
        
        # Train fold
        model, metrics, history = train_fold(
            model, train_loader, val_loader, config, device, fold_num=fold
        )
        
        # Get OOF predictions
        oof_results = evaluate(model, val_loader, device)
        all_oof_preds[val_idx] = oof_results['predictions']
        
        # fold_models.append(model)
        fold_metrics.append(metrics)
        
        print(f"\nFold {fold+1} Best Metrics:")
        print(f"  Spearman: {metrics['spearman']:.4f}")
        print(f"  Pearson:  {metrics['pearson']:.4f}")
        print(f"  Acc@Std:  {metrics['acc_std']:.4f}")
        
        # Save fold model
        model_path = Path(output_dir) / f"model_fold_{fold+1}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  Saved to: {model_path}")
        
        # Clear memory
        del model
        import gc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate overall OOF metrics
    oof_gold = full_df['average'].values
    oof_stds = full_df['stdev'].values
    
    oof_spearman = spearmanr(all_oof_preds, oof_gold)[0]
    oof_pearson = pearsonr(all_oof_preds, oof_gold)[0]
    
    oof_correct = sum(1 for p, g, s in zip(all_oof_preds, oof_gold, oof_stds) 
                     if abs(p - g) <= max(s, 1.0))
    oof_acc = oof_correct / len(oof_gold)
    
    print("\n" + "=" * 60)
    print("OVERALL OUT-OF-FOLD RESULTS")
    print("=" * 60)
    print(f"OOF Spearman:  {oof_spearman:.4f}")
    print(f"OOF Pearson:   {oof_pearson:.4f}")
    print(f"OOF Acc@Std:   {oof_acc:.4f}")
    
    # Average fold metrics
    avg_spearman = np.mean([m['spearman'] for m in fold_metrics])
    avg_pearson = np.mean([m['pearson'] for m in fold_metrics])
    avg_acc = np.mean([m['acc_std'] for m in fold_metrics])
    
    print(f"\nAverage Fold Metrics:")
    print(f"Avg Spearman:  {avg_spearman:.4f} ± {np.std([m['spearman'] for m in fold_metrics]):.4f}")
    print(f"Avg Pearson:   {avg_pearson:.4f} ± {np.std([m['pearson'] for m in fold_metrics]):.4f}")
    print(f"Avg Acc@Std:   {avg_acc:.4f} ± {np.std([m['acc_std'] for m in fold_metrics]):.4f}")
    
    return fold_metrics, all_oof_preds


def main():
    parser = argparse.ArgumentParser(description="T5-v1 AmbiStory Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run single epoch with 1 fold for testing"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Dry run modifications
    if args.dry_run:
        config['training']['epochs'] = 1
        config['training']['n_folds'] = 1
        print("DRY RUN MODE: 1 epoch, 1 fold")
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device(
        config['hardware']['device'] 
        if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    output_dir = Path(config['output']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dir = Path(config['data']['data_dir'])
    train_path = data_dir / config['data']['train_file']
    dev_path = data_dir / config['data']['dev_file']
    
    print(f"\nLoading data from: {data_dir}")
    train_df = pd.read_json(train_path).T.reset_index(drop=True)
    dev_df = pd.read_json(dev_path).T.reset_index(drop=True)
    full_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    print(f"Total samples: {len(full_df)}")
    print(f"Unique homonyms: {full_df['homonym'].nunique()}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    special_tokens = get_special_tokens()
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir / "tokenizer")
    
    # Initialize augmenter
    augmenter = DataAugmenter(aug_prob=config['augmentation']['probability'])
    
    # Train
    fold_metrics, oof_preds = train_with_kfold(
        full_df, config, tokenizer, augmenter, device, output_dir
    )
    
    # Save config
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Save OOF predictions
    np.save(output_dir / "oof_predictions.npy", oof_preds)
    
    print(f"\nTraining complete! Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
