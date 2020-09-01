#!/bin/bash
#SBATCH --job-name=NeST_DrugCell
#SBATCH --output=out.log
#SBATCH --partition=gpu
#SBATCH --nodelist=nrnb-6-1
#SBATCH --gres=gpu:1
#SBATCH --dependency=singleton

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

bash "${homedir}/scripts/cv_train.sh" $homedir $1
bash "${homedir}/scripts/cv_test_gpu.sh" $homedir $1
