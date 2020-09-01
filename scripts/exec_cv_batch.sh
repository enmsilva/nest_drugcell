#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugCell"

for i in {1..5}
do
	sbatch ${homedir}/scripts/cv_batch.sh $i
done
