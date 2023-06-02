#!/bin/bash

#SBATCH --account=gol_jfr267_uksr               #Name of account to run under
#SBATCH --partition=V4V32_SKY32M192_L  #partition

#SBATCH --job-name=npp_evaluate  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=12G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1            # number of gpus
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)

date
$HOME/.conda/envs/npp_asr/bin/python evaluate.py "test_nochanges_12-21-22" 
$HOME/.conda/envs/npp_asr/bin/python evaluate.py "test_combtones_12-21-22" 
$HOME/.conda/envs/npp_asr/bin/python evaluate.py "test_combdiac_12-21-22" 
$HOME/.conda/envs/npp_asr/bin/python evaluate.py "test_combboth_12-21-22" 
date