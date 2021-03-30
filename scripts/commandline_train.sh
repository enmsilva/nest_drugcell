#!/bin/bash
homedir=$1
gene2idfile="${homedir}/data/gene2ind_clinical_trial.txt"
cell2idfile="${homedir}/data/cell2ind.txt"
drug2idfile="${homedir}/data/drug2ind.txt"
ontfile="${homedir}/data/NeST_clinical_trial_ont.txt"
mutationfile="${homedir}/data/cell2mutation_clinical_trial.txt"
drugfile="${homedir}/data/drug2fingerprint.txt"
traindatafile="${homedir}/data/1_drugcell_test.txt"

modeldir="${homedir}/models"
if [ -d $modeldir ]
then
        rm -rf $modeldir
fi
mkdir -p $modeldir

cudaid=0

pyScript="${homedir}/src/train_drugcell.py"

source activate pytorch3drugcell

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile \
	-cell2id $cell2idfile -train $traindatafile -genotype $mutationfile -fingerprint $drugfile \
	-genotype_hiddens 10 -drug_hiddens '100,50,6' -final_hiddens 6 -model $modeldir \
	-cuda $cudaid -batchsize 5000 -epoch 50 -optimize 1 > train.log
