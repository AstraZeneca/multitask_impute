#!/bin/bash
#SBATCH -N 20
#SBATCH -c 1
#SBATCH --mem-per-cpu=196G
#SBATCH --time=0-12:00:00 # 12 hours
#SBATCH --output=logfiles/train_imputation_models.txt
#SBATCH --job-name="train_imputation_models"
#SBATCH --mail-user=
#SBATCH --mail-type=END,FAIL

# Put commands for executing job below this line
source tdimpute_env2/bin/activate
python TDimpute/TDimpute_for_omiembed.py
