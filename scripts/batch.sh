#!/bin/bash
#SBATCH --job-name=NeST_DrugCell
#SBATCH --output=out.log
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --dependency=singleton

bash "${1}/scripts/train.sh" $1 $2 $3
bash "${1}/scripts/test_genie.sh" $1 $2 $3
