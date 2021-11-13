#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:2
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 train_powder_deterministic1.py

./gpua.out
