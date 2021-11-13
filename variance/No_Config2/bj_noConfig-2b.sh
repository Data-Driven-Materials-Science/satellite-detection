#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-small
#SBATCH -t 8:00:00
#SBATCH --gpus=2
python3 noConfig-2b.py

./gpua.out
