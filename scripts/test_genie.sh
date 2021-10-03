#!/bin/bash
homedir=$1
gene2idfile="${homedir}/data/gene2ind_${2}.txt"
cell2idfile="${homedir}/data/GENIE/GENIE_all_cell2ind.txt"
drug2idfile="${homedir}/data/GENIE/GENIE_all_drug2ind.txt"
mutationfile="${homedir}/data/GENIE/GENIE_cell2mutation_${2}.txt"
drugfile="${homedir}/data/GENIE/GENIE_all_drug2fingerprint.txt"
testdatafile="${homedir}/data/GENIE/GENIE_test_zscore.txt"
zscore_method=$3

modeldir="${homedir}/model_${2}_${4}_${3}_${5}"
modelfile="${modeldir}/model_final.pt"

stdfile="${modeldir}/std.txt"

resultfile="${modeldir}/predict_genie"

hiddendir="${modeldir}/hidden_genie"
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
