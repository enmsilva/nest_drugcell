#!/bin/bash
#SBATCH --job-name=RLIPP_NeST_DrugCell
#SBATCH --output=cpu_out.log
#SBATCH --partition=nrnb-compute
#SBATCH --nodelist=nrnb-cn-12
#SBATCH --mem=64G

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

bash "${homedir}/scripts/rlipp_exec.sh" $homedir

