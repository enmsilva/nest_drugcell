#!/bin/bash

homedir=$1
hidden="${homedir}/rlipp/hidden"
ontology="${homedir}/data/ontology_${2}.txt"
drug_index="${homedir}/data/drug2ind.txt"
gene_index="${homedir}/data/gene2ind_${2}.txt"
cell_index="${homedir}/data/cell2ind.txt"
cell_mutation="${homedir}/data/cell2mutation_${2}.txt"
output="${homedir}/result/rlipp.out"

test="${homedir}/data/drugcell_all.txt"
predicted="${homedir}/result/drugcell_all.predict"

rlippdir="${homedir}/rlipp"
mkdir -p $rlippdir

if [ ! -d $hidden ]
then
	mkdir $hidden
	cp -r "${homedir}/model_${2}/hidden/*" "${hidden}/"
fi


python -u "${homedir}/src/rlipp_helper.py" -hidden $hidden -ontology $ontology \
	-drug_index $drug_index -gene_index $gene_index -cell_index $cell_index \
	-cell_mutation $cell_mutation -output $output -test $test -predicted $predicted -drug_count 0 -genotype_hiddens 6

