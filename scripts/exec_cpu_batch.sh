#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

for network in clinical_trial
do
	sbatch ${homedir}/scripts/cpu_batch.sh $network
done
