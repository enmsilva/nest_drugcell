#!/bin/bash

homedir=$1
drug_index="${homedir}/data/drug2ind.txt"
gene_index="${homedir}/data/gene2ind.txt"
cell_index="${homedir}/data/cell2ind.txt"
cell_mutation="${homedir}/data/cell2mutation.txt"
drug_fingerprint="${homedir}/data/drug2fingerprint.txt"
output="${homedir}/result/elastic_net.predict"
test="${homedir}/data/drugcell_all.txt"

python -u ${homedir}/src/elastic_net.py -drug_index $drug_index -gene_index $gene_index -cell_index $cell_index \
	-cell_mutation $cell_mutation -drug_fingerprint $drug_fingerprint -output $output -test $test

