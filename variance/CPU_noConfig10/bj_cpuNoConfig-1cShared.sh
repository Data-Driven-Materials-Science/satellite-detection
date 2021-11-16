#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=2

python3 CPU_No-Config-1cShared.py

./gpua.out
