#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

strict_param=$1
zscore_method=$2

for ontology in cg
do
	for i in 41
	do
		sbatch --job-name "NDC_${ontology}_${strict_param}_${i}" --output "out_${ontology}_${strict_param}_${i}.log" ${homedir}/scripts/strict_cv_batch.sh $homedir $ontology $i ${strict_param} ${zscore_method}
	done
done
