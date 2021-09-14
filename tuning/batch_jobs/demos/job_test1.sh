#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1

echo "Hello World"


./gpua.out

