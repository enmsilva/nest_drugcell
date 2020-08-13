#!/bin/bash
homedir=$1
gene2idfile="${homedir}/data/gene2ind.txt"
cell2idfile="${homedir}/data/cell2ind.txt"
drug2idfile="${homedir}/data/drug2ind.txt"
ontfile="${homedir}/data/drugcell_ont.txt"
mutationfile="${homedir}/data/cell2mutation.txt"
drugfile="${homedir}/data/drug2fingerprint.txt"
traindatafile="${homedir}/data/drugcell_train.txt"

modeldir="${homedir}/models"
if [ -d $modeldir ]
then
        rm -rf $modeldir
	mkdir $modeldir
fi

cudaid=$2

pyScript="${homedir}/code/cross_validate.py"

source activate pytorch3drugcell

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile \
	-cell2id $cell2idfile -train $traindatafile -model $modeldir -cuda $cudaid \
	-genotype $mutationfile -fingerprint $drugfile -genotype_hiddens 6 \
	-drug_hiddens '100,50,6' -final_hiddens 6 -batchsize 5000 > train.log
