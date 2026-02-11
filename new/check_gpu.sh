#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:H100:1
#SBATCH --output=check_gpu.out
#SBATCH --error=check_gpu.err

nvidia-smi