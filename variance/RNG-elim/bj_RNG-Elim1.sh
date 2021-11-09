#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-small
#SBATCH -t 2:00:00
#SBATCH --gpus=1

python3 RNG_Eliminator_1.py

./gpua.out
