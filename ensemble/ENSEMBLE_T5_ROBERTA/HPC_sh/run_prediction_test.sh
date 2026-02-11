#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=ensemble_predict
#SBATCH --output=test.out
#SBATCH --error=test.err

# Proper conda initialization for SLURM
source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate newenv

# Safety check (prints env name)
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
python --version
conda list | grep -E 'torch|cuda'

# GPU diagnostics
echo "nvidia-smi output:"
nvidia-smi
echo "PyTorch CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "PyTorch CUDA version:"
python -c "import torch; print(torch.version.cuda)"

# Fix MKL issue
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

	cd /home/siba/SemEval_Task05/ENSEMBLE_T5_ROBERTA/

python predict.py test
