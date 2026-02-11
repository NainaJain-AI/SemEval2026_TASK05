# AmbiStory: Word Sense Plausibility Rating (T5-v1)

This repository implements a **State-of-the-Art System** for SemEval Task 5: AmbiStory. It leverages a large-scale Cross-Encoder architecture (`deberta-v3-large`) with specialized mechanisms for ordinal regression, designed specifically for High-Performance Computing (HPC) environments.

---

## 🏆 System Overview & Key Improvements

This system evolves significantly from the initial baseline (`Untitled0.ipynb`), moving from a simple regression script to a robust, production-grade pipeline.

| Component | Baseline (`Untitled0.ipynb`) | **T5-v1 System (This Repo)** | Why it matters |
| :--- | :--- | :--- | :--- |
| **Foundation Model** | `deberta-v3-base` (184M params) | **`deberta-v3-large` (434M params)** | Larger capacity captures subtle semantic nuances in stories. |
| **Input Representation** | Simple concatenation | **Structured Special Tokens** | `[STORY]`, `[AMBIGUOUS]`, `[SENSE]` tags guide the model's attention explicitly. |
| **Pooling Strategy** | `[CLS]` token embedding | **Attention-Weighted Pooling** | Learns to weigh all words in the sequence, not just the start token. |
| **Prediction Head** | Linear Layer | **Multi-Layer Perceptron (MLP)** | Non-linear regression head with GELU activations for complex mapping. |
| **Loss Function** | MSE (Mean Squared Error) | **Ordinal Regression Loss** | Treats rating as a ranking problem (1 < 2 < 3...), superior for human-annotated scores. |
| **Training Scheme** | Single Train/Val split | **5-Fold Cross-Validation** | Prevents overfitting and provides robust error estimation. |
| **Infrastructure** | Notebook-based | **HPC / SLURM Optimized** | Handles OOM, Gradient Checkpointing, and Distributed Data Loading. |

---

## 🧠 Technical Deep Dive

### 1. Data Preprocessing & Special Tokens
Instead of raw text, we structure inputs to force the model to understand the role of each sentence.
*   **Input Format**:
    ```text
    [STORY] <Pre-Context> [AMBIGUOUS] <Sentence with Target> [ENDING] <Ending> [HOMONYM] <Word> [SENSE] <Definition> [EXAMPLE] <Usage>
    ```
*   **Target Marking**: The ambiguous word in the sentence is wrapped in `[TGT] ... [/TGT]` tags (e.g., `[TGT] bank [/TGT]`) to focus the attention mechanism.

### 2. Model Architecture (`models/regressor.py`)
We use `microsoft/deberta-v3-large` as the backbone.
*   **Encoder**: Outputs hidden states for all tokens.
*   **Attention Pooling**: Instead of just taking the first token (`CLS`), we compute a learned weighted average of *all* token embeddings. This captures context from the "Definition" and "Story" parts equally.
*   **Regression Head**: A 2-layer MLP with `LayerNorm` and `GELU` activation projects the pooled embedding to a single scalar score (1.0 - 5.0).

### 3. Loss Function (`models/losses.py`)
We employ a hybrid approach:
*   **BCEWithLogitsLoss**: We treat the regression as a classification task over thresholds (Is score > 1? Is score > 2? ...). This respects the ordinal nature of Likert scales.
*   Recall that simple MSE assumes the difference between 1-2 is semantically the same as 3-4, which isn't always true in NLP ratings. Ordinal regression fixes this.

---

## 📂 Repository Structure

```
T5-v1/
├── configs/
│   └── model_config.yaml  # Master control: Batch size, LR, Paths
├── data/                  # Place train.json / test.json here
├── HPC_sh/                # SLURM Execution Scripts
│   ├── run_training.sh        # Trains 5 folds (takes ~4-6 hours)
│   ├── run_prediction_dev.sh  # Validates on Dev set
│   └── run_prediction_test.sh # Generates Submission file
├── models/
│   ├── regressor.py       # Neural Network Architecture
│   └── losses.py          # Custom Loss Functions
├── utils/
│   └── data_loader.py     # Tokenization & Augmentation logic
├── train.py               # Main training loop (K-Fold)
├── predict.py             # Inference & Ensembling logic
└── evaluate.py            # Local scoring (Spearman/Pearson)
```

---

## 💻 HPC Execution Guide

This code is optimized for **NVIDIA H100/A100** nodes running SLURM.

### 1. Setup & Training
```bash
# 1. Pull latest code
git pull origin main

# 2. Submit Training Job (Background)
sbatch HPC_sh/run_training.sh
```
*   **What happens**: Trains 5 models on 5 splits of data.
*   **Monitoring**: `tail -f HPC_sh/HPC_logs/Train_stdout.log`

### 2. Validation (Dev Set)
Once training finishes, generate scores for the development set to check accuracy.
```bash
sbatch HPC_sh/run_prediction_dev.sh
```
*   **Target Metric**: Look for **Spearman Correlation** > 0.70 in the logs.

### 3. Submission (Blind Test Set)
Generates the final `predictions_test.json` for the leaderboard.
```bash
sbatch HPC_sh/run_prediction_test.sh
```
*   **Note**: This script automatically handles hidden labels (e.g., `(???)`) by using dummy placeholders during inference.

---

## 🔧 Troubleshooting & Config

### Common HPC Issues
| Error | Cause | Fix |
| :--- | :--- | :--- |
| **CUDA OOM / Job Cancelled** | Model too big for VRAM. | We already fixed this by using **Gradient Checkpointing** and specific **Shard** requests (24GB+). Do not increase batch size > 4 without more VRAM. |
| **System OOM (Killed)** | Too many data workers. | Keep `num_workers: 2` in `model_config.yaml`. |
| **ValueError: could not convert string to float** | Hidden labels in Test set. | Fixed using try/except block in `data_loader.py`. |

### Hyperparameters (`configs/model_config.yaml`)
*   **Batch Size**: 4 (Effective 32 via Accumulation Steps = 8).
*   **Learning Rate**: 1e-5 (Lower is better for Large models).
*   **Epochs**: 8 (with Early Stopping).
