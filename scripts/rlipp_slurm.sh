#!/bin/bash
#SBATCH --job-name=NeST_DrugCell_RLIPP
#SBATCH --output=cpu_out.log
#SBATCH --partition=nrnb-compute
#SBATCH --nodelist=nrnb-cn-02
#SBATCH --account=nrnb
#SBATCH --mem=256G
#SBATCH --cpus-per-task=60
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00

cpu_count=60

bash "${1}/scripts/rlipp.sh" $1 $2 $cpu_count
