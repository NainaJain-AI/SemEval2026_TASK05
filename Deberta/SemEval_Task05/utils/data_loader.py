"""
Data loading utilities for T5-v1 AmbiStory model.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmenter import DataAugmenter


# Global normalization statistics (computed from training data)
LABEL_MEAN = 3.1382
LABEL_STD = 1.1912


def safe_str(x) -> str:
    """Convert to string, handling None and NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def build_context(row: Dict) -> str:
    """
    Build context with explicit homonym marking and structural tokens.
    
    This helps the model understand which word is ambiguous.
    
    Args:
        row: Sample dictionary with precontext, sentence, ending, homonym
        
    Returns:
        Structured context string
    """
    homonym = safe_str(row.get("homonym", ""))
    precontext = safe_str(row.get("precontext", ""))
    sentence = safe_str(row.get("sentence", ""))
    ending = safe_str(row.get("ending", ""))
    
    # Mark the homonym in the sentence for better attention
    if homonym and homonym.lower() in sentence.lower():
        # Add markers around the homonym
        pattern = re.compile(re.escape(homonym), re.IGNORECASE)
        sentence = pattern.sub(f"[TGT] {homonym} [/TGT]", sentence, count=1)
    
    context = f"[STORY] {precontext} [AMBIGUOUS] {sentence}"
    
    if ending:
        context += f" [ENDING] {ending}"
    else:
        context += " [ENDING] None"
    
    context += f" [HOMONYM] {homonym}"
    
    return context.strip()


def build_sense(row: Dict) -> str:
    """
    Build sense representation with clear structure.
    
    Combines the dictionary definition with an example sentence.
    
    Args:
        row: Sample dictionary with judged_meaning, example_sentence
        
    Returns:
        Structured sense string
    """
    meaning = safe_str(row.get("judged_meaning", ""))
    example = safe_str(row.get("example_sentence", ""))
    
    sense = f"[SENSE] {meaning}"
    if example:
        sense += f" [EXAMPLE] {example}"
    
    return sense.strip()


class SemEvalDataset(Dataset):
    """
    PyTorch Dataset for SemEval 2026 Task 5 AmbiStory.
    
    Features:
    - Improved text representation with structural tokens
    - On-the-fly data augmentation
    - Support for training/inference modes
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = 384,
        augmenter: Optional[DataAugmenter] = None,
        is_training: bool = True,
        label_mean: float = LABEL_MEAN,
        label_std: float = LABEL_STD
    ):
        """
        Args:
            df: DataFrame with samples
            tokenizer: HuggingFace tokenizer
            max_len: Maximum sequence length
            augmenter: DataAugmenter instance for training
            is_training: Whether this is for training (enables augmentation)
            label_mean: Mean for label normalization
            label_std: Std for label normalization
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augmenter = augmenter
        self.is_training = is_training
        self.label_mean = label_mean
        self.label_std = label_std
        
        for _, row in df.iterrows():
            try:
                raw_label = float(row.get("average", 3.0))
                std = float(row.get("stdev", 1.0))
            except (ValueError, TypeError):
                # Handle test data with hidden labels (e.g. "(???)")
                raw_label = 3.0
                std = 1.0

            sample = {
                'row': row.to_dict(),
                'raw_label': raw_label,
                'std': std,
            }
            # Normalize label
            sample['label'] = (sample['raw_label'] - label_mean) / label_std
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        row = sample['row'].copy()
        
        # Apply data augmentation during training
        if self.is_training and self.augmenter and self.augmenter.should_augment():
            row = self.augmenter.augment_sample(row, method='synonym')
        
        # Build text representations
        context = build_context(row)
        sense = build_sense(row)
        
        # Tokenize with pair encoding
        enc = self.tokenizer(
            context,
            sense,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(sample['label'], dtype=torch.float),
            "std": torch.tensor(sample['std'], dtype=torch.float),
            "raw_label": torch.tensor(sample['raw_label'], dtype=torch.float)
        }


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load AmbiStory data from JSON file.
    
    Args:
        data_path: Path to JSON file
        
    Returns:
        DataFrame with samples
    """
    df = pd.read_json(data_path).T.reset_index(drop=True)
    return df


def compute_label_statistics(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute mean and std for label normalization.
    
    Args:
        df: Training DataFrame
        
    Returns:
        Tuple of (mean, std)
    """
    label_mean = df["average"].mean()
    label_std = df["average"].std()
    return label_mean, label_std


def get_special_tokens() -> Dict[str, List[str]]:
    """Get special tokens for the tokenizer."""
    return {
        'additional_special_tokens': [
            '[STORY]', '[AMBIGUOUS]', '[ENDING]', '[HOMONYM]',
            '[SENSE]', '[EXAMPLE]', '[TGT]', '[/TGT]'
        ]
    }
