"""
Evaluation script for T5-v1 AmbiStory model.

Usage:
    python evaluate.py --predictions_file predictions.json --gold_file dev.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def evaluate_predictions(predictions_file: str, gold_file: str) -> dict:
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions_file: Path to predictions JSON file
        gold_file: Path to gold labels JSON file
        
    Returns:
        Dictionary of metrics
    """
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Load gold labels
    gold_df = pd.read_json(gold_file).T.reset_index(drop=True)
    
    # Align predictions with gold
    y_pred = []
    y_true = []
    y_std = []
    
    for _, row in gold_df.iterrows():
        sample_id = str(row.get('sample_id', row.name))
        
        if sample_id in predictions:
            y_pred.append(float(predictions[sample_id]))
            y_true.append(float(row['average']))
            y_std.append(float(row.get('stdev', 1.0)))
        else:
            print(f"Warning: Missing prediction for sample {sample_id}")
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_std = np.array(y_std)
    
    # Calculate metrics
    pearson = pearsonr(y_pred, y_true)[0]
    spearman = spearmanr(y_pred, y_true)[0]
    
    # Accuracy within standard deviation (at least 1)
    correct = sum(1 for p, g, s in zip(y_pred, y_true, y_std) 
                  if abs(p - g) <= max(s, 1.0))
    acc_std = correct / len(y_pred)
    
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Spearman Correlation:':<25} {spearman:>10.4f}")
    print(f"{'Pearson Correlation:':<25} {pearson:>10.4f}")
    print(f"{'Accuracy Within StDev:':<25} {acc_std:>10.4f}")
    print(f"{'Mean Absolute Error:':<25} {mae:>10.4f}")
    print(f"{'Root Mean Square Error:':<25} {rmse:>10.4f}")
    print("-" * 40)
    print(f"{'Number of Examples:':<25} {len(y_pred):>10}")
    print("=" * 60)
    
    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    errors = y_pred - y_true
    
    print(f"\nPrediction Distribution:")
    print(f"  Range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"  Mean:  {y_pred.mean():.2f}")
    print(f"  Std:   {y_pred.std():.2f}")
    
    print(f"\nGold Distribution:")
    print(f"  Range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"  Mean:  {y_true.mean():.2f}")
    print(f"  Std:   {y_true.std():.2f}")
    
    print(f"\nError Distribution:")
    print(f"  Mean:  {errors.mean():.4f}")
    print(f"  Std:   {errors.std():.4f}")
    print(f"  |Error| < 0.5: {(np.abs(errors) < 0.5).sum()} ({100*(np.abs(errors) < 0.5).mean():.1f}%)")
    print(f"  |Error| < 1.0: {(np.abs(errors) < 1.0).sum()} ({100*(np.abs(errors) < 1.0).mean():.1f}%)")
    print(f"  |Error| > 2.0: {(np.abs(errors) > 2.0).sum()} ({100*(np.abs(errors) > 2.0).mean():.1f}%)")
    
    return {
        'spearman': spearman,
        'pearson': pearson,
        'acc_std': acc_std,
        'mae': mae,
        'rmse': rmse,
        'num_examples': len(y_pred)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate T5-v1 predictions")
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--gold_file",
        type=str,
        required=True,
        help="Path to gold labels JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional output file for metrics JSON"
    )
    
    args = parser.parse_args()
    
    metrics = evaluate_predictions(args.predictions_file, args.gold_file)
    
    # Optionally save metrics
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output_file}")


if __name__ == "__main__":
    main()
