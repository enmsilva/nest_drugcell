#!/bin/bash
homedir=$1
gene2idfile="${homedir}/data/gene2ind_${2}.txt"
cell2idfile="${homedir}/data/cell2ind_cg.txt"
drug2idfile="${homedir}/data/drug2ind_cg.txt"
mutationfile="${homedir}/data/cell2mutation_${2}.txt"
drugfile="${homedir}/data/drug2fingerprint_cg.txt"
testdatafile="${homedir}/data/${3}_test_cg.txt"
zscore_method=$4

modeldir="${homedir}/model_${2}_${3}_${4}"
modelfile="${modeldir}/model_final.pt"

stdfile="${modeldir}/std.txt"

resultfile="${modeldir}/predict"

hiddendir="${modeldir}/hidden"
if [ -d $hiddendir ]
then
	rm -rf $hiddendir
fi
mkdir -p $hiddendir

cudaid=0

pyScript="${homedir}/src/predict_drugcell.py"

source activate cuda11_env

python -u $pyScript -gene2id $gene2idfile -cell2id $cell2idfile -drug2id $drug2idfile \
	-genotype $mutationfile -fingerprint $drugfile -std $stdfile -hidden $hiddendir -result $resultfile \
	-batchsize 20000 -predict $testdatafile -zscore_method $zscore_method -load $modelfile -cuda $cudaid > "${modeldir}/test.log"
