#!/bin/bash
#SBATCH --job-name=NeST_DrugCell_${2}
#SBATCH --output=out_${2}.log
#SBATCH --partition=nrnb-gpu
#SBATCH --nodelist=nrnb-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --dependency=singleton

bash "${1}/scripts/commandline_train.sh" $1 $2
bash "${1}/scripts/commandline_test_gpu.sh" $1 $2
