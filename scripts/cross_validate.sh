#!/bin/bash

cudaid=1

homedir="/cellar/users/asinghal/Workspace/DrugCell"
dataFile="${homedir}/data/drugcell_all.txt"
trainFile="drugcell_train.txt"
testFile="drugcell_test.txt"


lc=`cat $dataFile | wc -l`

for i in {5..5}
do

	iterdir="${homedir}/iter$i"
	if [ ! -d $iterdir ]
	then	
		mkdir $iterdir

		min=$(( $lc * ($i - 1) / 5 ))
		max=$(( $lc * $i / 5 ))

		sed -n "${min},${max}p" $dataFile > "${iterdir}/${testFile}"
		if [[ $min > 1 ]]
		then
			min=$(( $min - 1 ))
			sed -n "1,${min}p" $dataFile >> "${iterdir}/${trainFile}"
		fi
		if [[ $max < $lc ]]
		then
			max=$(( $max + 1))
			sed -n "${max},${lc}p" $dataFile >> "${iterdir}/${trainFile}"
		fi
	fi
	#cudaid=$(( $i - 2 ))
	sbatch --output="${iterdir}/out.log" batch.sh $cudaid $i

done
