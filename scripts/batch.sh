#!/bin/bash
#SBATCH --job-name=NeST_DrugCell
#SBATCH --output=out.log
#SBATCH --partition=gpu
#SBATCH --nodelist=nrnb-6-1
#SBATCH --gres=gpu:1
#SBATCH --dependency=singleton

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

bash "${homedir}/scripts/commandline_train.sh" $homedir
bash "${homedir}/scripts/commandline_test_gpu.sh" $homedir
