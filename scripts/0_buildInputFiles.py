import sys
import os
from collections import defaultdict

import argparse


def printHidden(inputdir, filename, datapoints, prediction, drug_hidden_file):
	outputfile = inputdir + '/' + filename + '.built'

	with open(inputdir + '/' + filename, 'r') as fi, open(inputdir + '/' + drug_hidden_file, 'r') as fd, open(outputfile, 'w') as fo:
		i = 0
		paircounts = {}

		for line in fi:
			drug_line = fd.readline().strip()

			pair = (datapoints[i][0], datapoints[i][1])
			value = datapoints[i][2]

			try:
				paircounts[pair] += 1
				
			except KeyError:
				paircounts[pair] = 1

			fo.write("%s$%s$%d\t%s %s\t%s\t%s\n" % (pair[0], pair[1], paircounts[pair], line.strip(), drug_line, prediction[i], value)) 
			i += 1


def loadMapping(filename):
	data = {}
	with open(filename, 'r') as fi:
		for line in fi:
			tokens = line.strip().split('\t')
			data[tokens[1]] = tokens[0]
	return data



def main(opt):
	data_file = opt.data
	result_file = opt.result
	hidden_dir = opt.hidden
	drug_hidden_file = opt.drughidden

	# load mapping
	cell2ind = loadMapping(opt.cell2id)
	drug2ind = loadMapping(opt.drug2id)

	# load input data file: data points are (cell, drug, drug_response)
	datapoints = []
	with open(data_file, 'r') as fi:
		for line in fi:
			tokens = line.strip().split()
			datapoints.append((tokens[0], drug2ind[tokens[1]], tokens[2]))

	prediction = []
	with open(result_file, 'r') as fi:
		for line in fi:
			prediction.append(line.strip())


	# build feature vector file
	for filename in os.listdir(hidden_dir):
		if filename.endswith(".hidden") == True and filename.startswith("GO") == True:
			printHidden(hidden_dir, filename, datapoints, prediction, drug_hidden_file)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='RLIPP calculation')
	parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default="/data/cellardata/drug_study/DrugCell/input_data/data/bp/cell_mutations_bp_cell2id.txt")
	parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default="/data/cellardata/drug_study/DrugCell/input_data/data/all/drug_fingerprints_drug2id.txt")
	parser.add_argument('-hidden', help='Input directory: where hidden files are', type=str)
	parser.add_argument('-data', help='File containing all data points', type=str, default="/data/cellardata/drug_study/DrugCell/input_data/5folds/drugcell_all.txt")
	parser.add_argument('-result', help='Prediction result file', type=str, default="GO:0008150.predict")
	parser.add_argument('-drughidden', help='Hidden file from the final drug layer', type=str, default="drug_3.hidden")

	opt = parser.parse_args()
	main(opt)



