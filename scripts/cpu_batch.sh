#!/bin/bash
#SBATCH --job-name=RLIPP_NeST_DrugCell
#SBATCH --output=out.log
#SBATCH --nodelist=nrnb-5-0
#SBATCH --dependency=singleton

homedir="/cellar/users/asinghal/Workspace/nest_drugcell/rlipp"

Rscript ${homedir}/rlipp.R  ${homedir} ${homedir}/drugcell_ont.txt ${homedir}/drugcell.predict 
