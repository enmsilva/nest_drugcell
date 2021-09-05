#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

for ontology in cg
do
	for i in {1..5}
	do
		sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/cv_batch.sh $homedir $ontology $i
	done
done
