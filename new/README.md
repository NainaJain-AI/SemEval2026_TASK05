# LLAMA Few-Shot Learning Model

This folder contains the LLAMA-based model with few-shot learning for semantic meaning evaluation.

## Structure

```
LLAMA/
├── configs/
│   └── model_config.yaml          # Model and training configuration
├── data/
│   ├── train.json                 # Training data
│   ├── dev.json                   # Development data
│   └── test.json                  # Test data
├── utils/
│   └── data_loader.py             # Few-shot dataset loader
├── models/                        # Saved model checkpoints
├── checkpoints/                   # Model weights storage
├── HPC_sh/
│   ├── run_training.sh            # SLURM script for training
│   ├── run_prediction_dev.sh       # SLURM script for dev predictions
│   └── run_prediction_test.sh      # SLURM script for test predictions
├── train.py                       # Training script with few-shot learning
├── predict.py                     # Prediction script
└── evaluate.py                    # Evaluation script
```

## Features

- **Few-Shot Learning**: Uses in-context learning with examples from training data
- **LLAMA-2-7b-hf**: Leverages the Llama 2 model for semantic understanding
- **Flexible Configuration**: All hyperparameters in YAML config file
- **GPU Support**: Optimized for H100 GPUs with mixed precision training

## Quick Start

1. **Training**:
   ```bash
   sbatch HPC_sh/run_training.sh
   # Or locally:
   python train.py
   ```

2. **Prediction**:
   ```bash
   # For dev set:
   sbatch HPC_sh/run_prediction_dev.sh
   # For test set:
   sbatch HPC_sh/run_prediction_test.sh
   ```

3. **Evaluation**:
   ```bash
   python evaluate.py
   ```

## Configuration

Edit `configs/model_config.yaml` to adjust:
- Model name
- Learning rates
- Batch size
- Number of few-shot examples
- Training epochs

## Few-Shot Learning Details

The model uses 5 examples from the training data as in-context demonstrations during both training and inference. This helps the model understand the task better without requiring extensive fine-tuning.

