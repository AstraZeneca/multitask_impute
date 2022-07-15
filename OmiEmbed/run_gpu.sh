#!/bin/bash
#SBATCH -p gpu                  # get gpu partition of resources
#SBATCH --gres=gpu:volta:2      # specify number and type of GPUs you want
#SBATCH --cpus-per-task=8       # cores per task
#SBATCH --nodes=1               # number of nodes to run on
#SBATCH --ntasks=1              # leave at 1 unless multiprocessing via MPI
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00       # job wall time, days-hours:minutes:seconds
#SBATCH --job-name="omiembed_gpu"
#SBATCH --output=output/multi_runs/prediction-%j.txt # outputs to user home directory by default
#SBATCH --mail-user=
#SBATCH --mail-type=END,FAIL
#SBATCH --constraint="cascadelake"

conda activate multi_venv
python train_test.py --deterministic --omics_mode abc --experiment_name rnadnamirna_imputed_multi_1_cv5 --cv_fold_index 5 --batch_size 16 --lr 0.001 --epoch_num_p1 50 --epoch_num_p2 50 --epoch_num_p3 10 --model vae_classifier
