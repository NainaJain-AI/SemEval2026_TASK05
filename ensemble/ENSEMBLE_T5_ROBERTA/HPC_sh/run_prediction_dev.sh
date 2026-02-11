#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --job-name=ensemble_train
#SBATCH --output=dev.out
#SBATCH --error=dev.err


# Proper conda initialization for SLURM
source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate newenv

# Safety check (prints env name)
echo "Using conda env: $CONDA_DEFAULT_ENV"
echo $CONDA_DEFAULT_ENV
which python
python -c "import torch; print(torch.__version__)"
# Fix MKL issue
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

cd /home/siba/SemEval_Task05/ENSEMBLE_T5_ROBERTA/

python predict.py dev
python evaluate.py
