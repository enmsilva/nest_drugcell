#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

for ontology in clinical_trial
do
	sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/batch.sh $homedir $ontology
done
