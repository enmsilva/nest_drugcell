#!/bin/bash
#SBATCH --job-name=NeST_DrugCell_EN
#SBATCH --output=cpu_out.log
#SBATCH --partition=nrnb-compute
#SBATCH --nodelist=nrnb-cn-08
#SBATCH --mem=64G

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

bash "${homedir}/scripts/elastic_net.sh" $homedir $1

