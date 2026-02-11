#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --job-name=ensemble_train
#SBATCH --output=train.out
#SBATCH --error=train.err

# Proper conda initialization for SLURM
echo "Initializing conda..."
source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate newenv
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
conda run -n myenv python --version || echo "Python not found!"

# Diagnostics: print job environment and resource info
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM Mem per node: $SLURM_MEM_PER_NODE"
echo "SLURM Partition: $SLURM_JOB_PARTITION"
echo "Hostname: $(hostname)"
echo "Start Time: $(date)"
free -h
df -h
ulimit -a

# Print Python path and version for debugging
echo "Python executable: $(which python)"
python --version || echo "Python not found in environment!"
conda info --envs

# Fix MKL issue
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1


cd /home/siba/SemEval_Task05/ENSEMBLE_T5_ROBERTA/
# Check GPU availability on the compute node
echo "Checking GPU availability on compute node:"
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
conda run -n myenv python train.py
