#!/bin/bash
#SBATCH --job-name=RLIPP_NeST_DrugCell
#SBATCH --output=cpu_out.log
#SBATCH --nodelist=nrnb-5-1
#SBATCH --mem=64G
#SBATCH --dependency=singleton

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

bash "${homedir}/scripts/rlipp_exec.sh" $homedir

