#!/bin/bash
#SBATCH --job-name=RLIPP_NeST_DrugCell
#SBATCH --output=cpu_out.log
#SBATCH --nodelist=nrnb-5-4
#SBATCH --mem=64G

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

bash "${homedir}/scripts/rlipp_exec.sh" $homedir

