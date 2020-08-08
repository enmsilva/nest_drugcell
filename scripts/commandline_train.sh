#!/bin/bash
homedir="/cellar/users/asinghal/Workspace/DrugCell"
gene2idfile="${homedir}/data/gene2ind.txt"
cell2idfile="${homedir}/data/cell2ind.txt"
drug2idfile="${homedir}/data/drug2ind.txt"
ontfile="${homedir}/data/drugcell_ont.txt"
mutationfile="${homedir}/data/cell2mutation.txt"
drugfile="${homedir}/data/drug2fingerprint.txt"

traindatafile="${homedir}/iter${2}/drugcell_train.txt"
valdatafile="${homedir}/iter${2}/drugcell_test.txt"

modeldir="${homedir}/models"
if [ -d $modeldir ]
then
        rm -rf $modeldir
	mkdir $modeldir
fi

cudaid=$1

pyScript="${homedir}/code/train_drugcell.py"

source activate pytorch3drugcell

echo "\n\nExecuting Iteration ${2}\n" >> train.log

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -test $valdatafile -model $modeldir -cuda $cudaid -genotype $mutationfile -fingerprint $drugfile -genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 -batchsize 5000 >> train.log

cp  "${modeldir}/model_final.pt"  "${homedir}/iter${2}/"
