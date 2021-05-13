#!/bin/bash
#SBATCH --job-name=NeST_DrugCell_RLIPP
#SBATCH --output=cpu_out.log
#SBATCH --partition=nrnb-compute
#SBATCH --nodelist=nrnb-cn-08
#SBATCH --mem=64G

bash "${1}/scripts/rlipp.sh" $1 $2

