#!/bin/bash
homedir=$1
gene2idfile="${homedir}/data/gene2ind_${2}.txt"
cell2idfile="${homedir}/data/GENIE/GENIE_cell2ind.txt"
drug2idfile="${homedir}/data/GENIE/GENIE_drug2ind.txt"
mutationfile="${homedir}/data/GENIE/GENIE_cell2mutation.txt"
drugfile="${homedir}/data/GENIE/GENIE_drug2fingerprint.txt"
testdatafile="${homedir}/data/GENIE/GENIE_test.txt"

modelfile="${homedir}/model_${2}/model_final.pt"

resultdir="${homedir}/result"
mkdir -p $resultdir

resultfile="${resultdir}/predict_genie_${2}"

hiddendir="${homedir}/hidden_genie"
if [ -d $hiddendir ]
then
	rm -rf $hiddendir
fi
mkdir -p $hiddendir

cudaid=0

pyScript="${homedir}/src/predict_drugcell.py"

source activate cuda11_env

python -u $pyScript -gene2id $gene2idfile -cell2id $cell2idfile -drug2id $drug2idfile \
	-genotype $mutationfile -fingerprint $drugfile -hidden $hiddendir -result $resultfile \
	-batchsize 10000 -predict $testdatafile -load $modelfile -cuda $cudaid > test.log
