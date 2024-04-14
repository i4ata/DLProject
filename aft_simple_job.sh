#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=aft_simple
#SBATCH --mem=8000

module purge

module load Python/3.9.6-GCCcore-11.2.0

source ~/env/bin/activate

python train_simple.py
