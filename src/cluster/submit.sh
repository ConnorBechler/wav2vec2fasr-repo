#!/bin/bash

#SBATCH --account=gol_jfr267_uksr               #Name of account to run under
#SBATCH --partition=V4V32_CAS40M192_L  #partition

#SBATCH --job-name=npp_finetune  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1            # number of gpus
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)

date
$HOME/.conda/envs/npp_asr/bin/python full_finetune.py #new_finetune.py
date