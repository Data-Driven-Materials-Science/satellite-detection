#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1

python3 train_powder_LR-0_001_T4.py

./gpua.out
