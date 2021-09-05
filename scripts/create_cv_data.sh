#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/nest_drugcell"
dataFile="${homedir}/data/drugcell_all_cg.txt"
trainFile="drugcell_train_cg.txt"
testFile="drugcell_test_cg.txt"

lc=`cat $dataFile | wc -l`

for i in {1..5}
do
	min=$(( ($lc * ($i - 1)) / 5 + 1 ))
	max=$(( ($lc * $i) / 5 ))

	sed -n "${min},${max}p" $dataFile > "${homedir}/data/${i}_${testFile}"
	if [[ $min > 1 ]]
	then
		min=$(( $min - 1 ))
		sed -n "1,${min}p" $dataFile >> "${homedir}/data/${i}_${trainFile}"
	fi
	if [[ $max < $lc ]]
	then
		max=$(( $max + 1))
		sed -n "${max},${lc}p" $dataFile >> "${homedir}/data/${i}_${trainFile}"
	fi
done
