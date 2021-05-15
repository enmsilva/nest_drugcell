#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/ddram_drugcell"

for ontology in ddram
do
	sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/batch.sh $homedir $ontology
done
