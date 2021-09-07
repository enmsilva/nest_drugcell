#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

strict_param=$1

for ontology in cg
do
	for i in {1..5}
	do
		sbatch --job-name "NDC_${ontology}_${strict_param}" --output "out_${ontology}_${strict_param}.log" ${homedir}/scripts/strict_cv_batch.sh $homedir $ontology $i ${strict_param}
	done
done
