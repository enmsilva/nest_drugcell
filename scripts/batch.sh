#!/bin/bash
#SBATCH --job-name=NeST_DrugCell
#SBATCH --output=out.log
#SBATCH --partition=gpu
#SBATCH --nodelist=nrnb-6-0
#SBATCH --dependency=singleton

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

bash "${homedir}/scripts/commandline_train.sh" $homedir $1
bash "${homedir}/scripts/commandline_test_gpu.sh" $homedir $1
