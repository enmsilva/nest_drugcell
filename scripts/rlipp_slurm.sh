#!/bin/bash
#SBATCH --job-name=NeST_DrugCell_RLIPP
#SBATCH --output=cpu_out.log
#SBATCH --partition=nrnb-compute
#SBATCH --nodelist=nrnb-cn-08
#SBATCH --mem=64G

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

bash "${homedir}/scripts/rlipp.sh" $homedir $1

