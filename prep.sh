#!/bin/bash

#SBATCH --account=gol_jfr267_uksr               #Name of account to run under
#SBATCH --partition=P4V12_SKY32M192_L  #partition

#SBATCH --job-name=npp_dataprep  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=12G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

date
#Data prep
$HOME/.conda/envs/npp_asr/bin/python data_preprocessing/data_process.py
$HOME/.conda/envs/npp_asr/bin/python data_preprocessing/vocab_setup.py

#Test run
# $HOME/.conda/envs/npp_asr/bin/python test.py
date