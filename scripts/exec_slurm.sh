#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

zscore_method=$1

for ontology in cg
do
	sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/batch.sh $homedir $ontology ${zscore_method}
	#sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/rlipp_slurm.sh $homedir $ontology
done
