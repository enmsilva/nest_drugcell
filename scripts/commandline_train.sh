#!/bin/bash
homedir=$1
gene2idfile="${homedir}/data/gene2ind_${2}.txt"
cell2idfile="${homedir}/data/cell2ind.txt"
drug2idfile="${homedir}/data/drug2ind.txt"
ontfile="${homedir}/data/ontology_${2}.txt"
mutationfile="${homedir}/data/cell2mutation_${2}.txt"
drugfile="${homedir}/data/drug2fingerprint.txt"
traindatafile="${homedir}/data/drugcell_all.txt"

modeldir="${homedir}/model_${2}"
if [ -d $modeldir ]
then
	rm -rf $modeldir
fi
mkdir -p $modeldir

cudaid=0

pyScript="${homedir}/src/train_drugcell.py"

source activate cuda11_env

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile \
	-cell2id $cell2idfile -train $traindatafile -genotype $mutationfile -fingerprint $drugfile \
	-genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 -model $modeldir \
	-cuda $cudaid -batchsize 10000 -epoch 300 -optimize 0 > train.log
