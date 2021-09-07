#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

#for ontology in cg cg_go fmg_718 fmg_718_go random_718_a random_718_b random_718_c random_718_d random_718_e cg_bb_a cg_bb_b cg_bb_c cg_bb_d cg_bb_e
do
	for i in {1..5}
	do
		sbatch --job-name "NDC_${ontology}" --output "out_${ontology}.log" ${homedir}/scripts/cv_batch.sh $homedir $ontology $i
	done
done
