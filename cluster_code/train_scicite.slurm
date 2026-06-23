#!/bin/bash
#SBATCH --job-name=lora_scicite
#SBATCH --partition=test
#SBATCH --gres=gpu:L40:1
#SBATCH --time=00:55:00
#SBATCH --output=slurm-%j.out

cd ~/citation-intent
source venv/bin/activate
export HF_HUB_OFFLINE=1
python train_scicite.py
