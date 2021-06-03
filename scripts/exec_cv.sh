#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

for ontology in clinical_trial_bb_d clinical_trial_bb_e
do
	for i in {1..5}
	do
		sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/cv_batch.sh $homedir $ontology $i
	done
done
