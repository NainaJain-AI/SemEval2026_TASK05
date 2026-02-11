#!/bin/bash
#SBATCH --job-name=T5v1_Train
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --gres=shard:24
#SBATCH --output=HPC_logs/T5v1_Train_output_%j.log
#SBATCH --error=HPC_logs/T5v1_Train_error_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=200
#SBATCH -t 12:00:00
#SBATCH --mem=64G

echo "=============================================="
echo "T5-v1 AmbiStory Training"
echo "=============================================="
echo "Start Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Navigate to project directory
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Create log directory
mkdir -p HPC_logs

# ============================================
# ENVIRONMENT SETUP
# ============================================
echo "Setting up environment..."

# 1. Try to load anaconda module (common on HPCs)
module load anaconda3/2023.03 2>/dev/null || module load python/3.10 2>/dev/null

# 2. Initialize conda
if [ -f "/apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh" ]; then
    source "/apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# 3. Activate environment
conda activate torch_gpu310 2>/dev/null || source activate torch_gpu310 2>/dev/null

# 4. Fallback for explicit path if conda activate fails
if [ $? -ne 0 ] && [ -f "$HOME/.conda/envs/torch_gpu310/bin/activate" ]; then
    source $HOME/.conda/envs/torch_gpu310/bin/activate torch_gpu310
fi

# 5. Fix CUDA library path issue (libnvJitLink.so.12 undefined symbol)
# Add the current environment's lib directory to LD_LIBRARY_PATH
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    echo "Added conda lib to LD_LIBRARY_PATH: $CONDA_PREFIX/lib"
fi

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo "WARNING: GPU NOT AVAILABLE OR PYTORCH NOT DETECTING IT!"
fi
echo ""

# ============================================
# DATA VERIFICATION
# ============================================
DATA_DIR="./data"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi
echo "Data directory found: $DATA_DIR"

# Run training
echo ""
echo "Starting training..."
PYTHONPATH=. python train.py \
    --config configs/model_config.yaml \
    2>&1 | tee HPC_logs/Train_stdout.log

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_CODE
