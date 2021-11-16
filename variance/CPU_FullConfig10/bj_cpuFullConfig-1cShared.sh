#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=2

CUBLAS_WORKSPACE_CONFIG=:16:8 python3 CPU_Full-Config-1cShared.py

./gpua.out
