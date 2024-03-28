#!/bin/bash

#SBATCH --job-name=nlp
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1
#SBATCH --nodelist=sh04
current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

source ../TP2/venvtp1/bin/activate

echo "Training"
python3 main.py --mode train

if [[ $? != 0 ]]; then
    exit -1
fi
