#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-small
#SBATCH -t 8:00:00
#SBATCH --gpus=2

CUBLAS_WORKSPACE_CONFIG=:16:8 python3 Full-Config-1c.py

./gpua.out
