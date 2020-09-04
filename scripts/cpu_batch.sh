#!/bin/bash
#SBATCH --job-name=RLIPP_NeST_DrugCell
#SBATCH --output=cpu_out.log
#SBATCH --nodelist=nrnb-5-0
#SBATCH --dependency=singleton

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

python -u "${homedir}/code/rlipp_analysis.py" $homedir

