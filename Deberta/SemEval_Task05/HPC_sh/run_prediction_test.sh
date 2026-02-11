#!/bin/bash
#SBATCH --job-name=T5v1_Pred_Test
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --gres=shard:12
#SBATCH --output=HPC_logs/T5v1_PredTest_output_%j.log
#SBATCH --error=HPC_logs/T5v1_PredTest_error_%j.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=200
#SBATCH -t 02:00:00
#SBATCH --mem=48G

echo "=============================================="
echo "T5-v1 AmbiStory - Test Set Prediction (Submission)"
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

# Environment Setup
echo "Setting up environment..."
module load anaconda3/2023.03 2>/dev/null || module load python/3.10 2>/dev/null

if [ -f "/apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh" ]; then
    source "/apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda activate torch_gpu310 2>/dev/null || source activate torch_gpu310 2>/dev/null

if [ $? -ne 0 ] && [ -f "$HOME/.conda/envs/torch_gpu310/bin/activate" ]; then
    source $HOME/.conda/envs/torch_gpu310/bin/activate torch_gpu310
fi

if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

echo "Python: $(which python)"
echo ""

# Define paths
TEST_FILE="./data/test.json"
MODEL_DIR="saved_models"
OUTPUT_FILE="predictions_test.json"

# Verify model exists
if [ ! -d "$MODEL_DIR" ] || ! ls $MODEL_DIR/model_fold_*.pt 1> /dev/null 2>&1; then
    echo "ERROR: No trained models found in $MODEL_DIR"
    echo "Please run training first!"
    exit 1
fi

echo "Models found: $(ls $MODEL_DIR/model_fold_*.pt | wc -l) folds"

# Verify test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    exit 1
fi

# Generate test predictions
echo ""
echo "Generating predictions on test set for submission..."
PYTHONPATH=. python predict.py \
    --config configs/model_config.yaml \
    --test_path "$TEST_FILE" \
    --output_file "$OUTPUT_FILE" \
    2>&1 | tee HPC_logs/PredTest_stdout.log

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "SUBMISSION FILE READY"
echo "=============================================="
echo "Output: $OUTPUT_FILE"
echo "Submit this file to CodaBench!"
echo ""
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=============================================="

exit $EXIT_CODE
