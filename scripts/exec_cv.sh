#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

for ontology in clinical_trial random bb
do
	for i in {1..5}
	do
		sbatch ${homedir}/scripts/cv_batch.sh $homedir $ontology $i
	done
done
