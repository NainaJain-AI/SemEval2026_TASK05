#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --job-name=llama_train
#SBATCH --output=train.out
#SBATCH --error=train.err

echo "Initializing conda..."
source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate newenv
echo "Conda env: $CONDA_DEFAULT_ENV"

# Job diagnostics
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM Partition: $SLURM_JOB_PARTITION"
echo "Hostname: $(hostname)"
echo "Start Time: $(date)"

# GPU check
echo "Checking GPU availability:"
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# Run training
cd /home/siba/SemEval_Task05/LLAMA/
export MKL_THREADING_LAYER=GNU

echo "Starting LLAMA training with few-shot learning..."
python train.py

