#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"

#Drugs

bash "$homedir/scripts/parse_ctrp_gdsc_auc_files.sh" $homedir 4 "$homedir/data/ctrp_gdsc_drug_list.txt" "$homedir/data/drug2ind_cg.txt"

bash "$homedir/scripts/parse_nci_auc_file.sh" $homedir 4 "$homedir/data/drug2ind_nci.txt"


#Cell lines

bash "$homedir/scripts/parse_ctrp_gdsc_auc_files.sh" $homedir 5 "$homedir/data/ctrp_gdsc_cell_list.txt" "$homedir/data/cell2ind_cg.txt"

bash "$homedir/scripts/parse_nci_auc_file.sh" $homedir 5 "$homedir/data/cell2ind_nci.txt"
