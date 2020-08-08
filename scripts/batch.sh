#!/bin/bash
#SBATCH --job-name=NeST_DrugCell
#SBATCH --output=out.log
#SBATCH --partition=gpu
#SBATCH --nodelist=nrnb-6-0
#SBATCH --dependency=singleton

bash commandline_train.sh $1 $2
bash commandline_test_gpu.sh $1 $2
