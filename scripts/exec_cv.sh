#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/ddram_drugcell"

for ontology in ddram ddram_bb_a ddram_bb_b ddram_bb_c ddram_bb_d ddram_bb_e
do
	for i in {1..5}
	do
		sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/cv_batch.sh $homedir $ontology $i
	done
done
