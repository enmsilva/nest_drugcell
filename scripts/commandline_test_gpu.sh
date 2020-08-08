#!/bin/bash
homedir="/cellar/users/asinghal/Workspace/DrugCell"
gene2idfile="${homedir}/data/gene2ind.txt"
cell2idfile="${homedir}/data/cell2ind.txt"
drug2idfile="${homedir}/data/drug2ind.txt"
mutationfile="${homedir}/data/cell2mutation.txt"
drugfile="${homedir}/data/drug2fingerprint.txt"

testdatafile="${homedir}/iter${2}/drugcell_test.txt"
modelfile="${homedir}/iter${2}/model_final.pt"
resultdir="${homedir}iter${2}/result"
mkdir -p $resultdir

hiddendir="${homedir}/hidden"
if [ -d $hiddendir ]
then
	rm -rf $hiddendir
	mkdir $hiddendir
fi

cudaid=$1

pyScript="${homedir}/code/predict_drugcell.py"

source activate pytorch3drugcell

echo "\n\nRunning Iteration ${2}\n" >> tests.log

python -u $pyScript -gene2id $gene2idfile -cell2id $cell2idfile -drug2id $drug2idfile -genotype $mutationfile -fingerprint $drugfile -hidden $hiddendir -result $resultdir -predict $testdatafile -load $modelfile -cuda $cudaid >> tests.log
