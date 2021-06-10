#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/ddram_drugcell"

for ontology in ddram
do
	sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/elastic_net_slurm.sh $homedir $ontology
done
