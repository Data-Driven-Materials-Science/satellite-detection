#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 16:00:00
#SBATCH --gpus=1
python3 noConfig-2b.py

./gpua.out
