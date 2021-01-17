#!/bin/bash

homedir=$1
hidden="${homedir}/rlipp/hidden/"
ontology="${homedir}/data/drugcell_ont.txt"
drug_index="${homedir}/data/drug2ind.txt"
gene_index="${homedir}/data/gene2ind.txt"
cell_index="${homedir}/data/cell2ind.txt"
cell_mutation="${homedir}/data/cell2mutation.txt"
output="${homedir}/result/rlipp.out"

test="${homedir}/data/drugcell_all.txt"
predicted="${homedir}/result/drugcell_all.predict"

rlippdir="${homedir}/rlipp"
mkdir -p $rlippdir

python -u ${homedir}/src/rlipp_helper.py -hidden $hidden -ontology $ontology \
	-drug_index $drug_index -gene_index $gene_index -cell_index $cell_index \
	-cell_mutation $cell_mutation -output $output -test $test -predicted $predicted -drug_count 50 -genotype_hiddens 10

