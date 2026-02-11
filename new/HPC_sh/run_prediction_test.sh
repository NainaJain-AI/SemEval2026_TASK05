#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --job-name=llama_predict_test
#SBATCH --output=test.out
#SBATCH --error=test.err

echo "Initializing conda..."
source /home/siba/anaconda3/etc/profile.d/conda.sh
conda activate newenv

echo "Running predictions on test split..."
cd /home/siba/SemEval_Task05/LLAMA/
export MKL_THREADING_LAYER=GNU

python predict.py test

