import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

import ddot
from ddot import Ontology
from multiprocessing import Pool

import argparse
from collections import defaultdict


def make_file_path(term):
	return input_dir + "/" + term + ".hidden.built"


def rlipp(parent, children, term2gene, cell2gene, linemask, mask=[], variance_explained=0.9):
	
	"""Calculates the relative improvement in performance

	See DCell original paper for details

	Parameters
	----------
	parent : str
		Term name
	children : list of str
		List of term names
	genes_map : dict
		Dictionary that maps term name to the genes that belong to that term
	variance_explained : float
		(0,1]. The number of principal components to include needed to reach 
		this threhsold.  
	"""

    # build input vectors for linear model for the parent
	df = pd.read_csv(make_file_path(parent), sep='\s+', index_col=0, header=None)
	hiddens_df = df.iloc[linemask, :] 

	genes = term2gene[parent].tolist()

	if len(mask) == 0:
		mask = check_index(hiddens_df.index, genes, cell2gene)

	print("processing: %s" % parent)
	print("number of samples = %s" % hiddens_df.shape[0])

	if hiddens_df.shape[0] == 0 or mask.count(False) == len(mask):
		return float("-inf")

	# build input vectors for linear model for the parent
	parent_pcs_df, parent_output_df, parent_mask = calculate_pcs(hiddens_df, mask=mask, variance_explained=variance_explained)
	
	# add new column names to the data frame
	parent_pcs_df = get_new_column_names(parent_pcs_df, name='%s$pc'%parent)
	parent_index = parent_pcs_df.index

	# build input vectors for linear model for the children
	children_pcs = []
	for child in children:

		child_df = pd.read_csv(make_file_path(child), sep='\s+', index_col=0, header=None)
		child_hiddens_df = child_df.iloc[linemask, :]

		print("processing a child: %s" % child)
		print("number of samples = %d" % child_hiddens_df.shape[0])

		child_pcs_df, _, _ = calculate_pcs(child_hiddens_df, mask=parent_mask, variance_explained=variance_explained)

		if not set(parent_index).issuperset(child_pcs_df.index):
			raise ValueError("Inconsistent parent and children indices! Expected children indices to be a subset of parent's")

		child_pcs_df = child_pcs_df.reindex(parent_index)
		children_pcs.append(child_pcs_df)

	children_pcs = pd.concat(children_pcs, axis=1).reindex(parent_index)
	parent_output_df = parent_output_df.reindex(parent_index)

	parent_score = calculate_performance(parent_pcs_df.values, parent_output_df.loc[:, 'Predicted'].values.ravel())
	children_score = calculate_performance(children_pcs, parent_output_df.loc[:, 'Predicted'].values.ravel())

	return parent_score/children_score



def rlipp_single(term, term2gene, cell2gene, gene2id, drug_vec_len, linemask, mask=[], variance_explained=0.9):
	print("processing (rlipp_single): %s" % term)

	genes = term2gene[term].tolist()

	# build input vectors for linear model for the parent
	df = pd.read_csv(make_file_path(term), sep='\s+', index_col=0, header=None)
	hiddens_df = df.iloc[linemask, :]

	# collect indices related to the term
	if len(mask) == 0:
		mask = check_index(hiddens_df.index, genes, cell2gene)

	if hiddens_df.shape[0] == 0 or mask.count(False) == len(mask):
		return float("-inf")


	# extract only the hidden weights related to the term
	hiddens_df = hiddens_df.loc[mask]

	hiddens = hiddens_df.iloc[:, :-2].values
	scaled_hiddens = scale(hiddens)
	scaled_hiddens = np.nan_to_num(scaled_hiddens)

	pca = PCA()
	pcs = pca.fit_transform(scaled_hiddens)

	n = len(pcs[0])
	print("number of neurons: %d" % n)

	if variance_explained < 1:
		n = get_number_of_pcs(pca.explained_variance_ratio_, variance_explained)

	if n < 2:
		n = 2

	print("number of PCs: %d" % n)

	# building input vectors for linear model for the genotype
	pcs_df = pd.DataFrame(pcs, index=hiddens_df.index)
	drug_feature = pcs_df.iloc[:, (-1*drug_vec_len)-2:-2]
	drug_feature.index = hiddens_df.index

	gene_feature = np.zeros((1, len(gene2id)))

	for gene in genes:
		gene_feature[0, gene2id[gene]] = 1

	gene_feature = pd.DataFrame(np.repeat(gene_feature, drug_feature.shape[0], axis=0), index=hiddens_df.index)
	print("drug_feature", drug_feature.shape)
	print("gene_feature", gene_feature.shape)

	children_pcs = pd.DataFrame(pd.concat([gene_feature, drug_feature], axis=1, join='inner'))
	print("children_feature", children_pcs.shape)

	# do PCA on parent feature
	hiddens = children_pcs.values
	scaled_hiddens = scale(hiddens)
	scaled_hiddens = np.nan_to_num(scaled_hiddens)

	children_pcs = pca.fit_transform(scaled_hiddens)

	parent_pcs_df = pd.DataFrame(pcs[:, :n], index=hiddens_df.index)
	output_df = hiddens_df.iloc[:, -2:]
	output_df.columns = ['Predicted', 'Measured']

	# do PCA on children feature
	children_pcs_df = pd.DataFrame(children_pcs[:, :n], index=hiddens_df.index)

	children_pcs_df = children_pcs_df.reindex(parent_pcs_df.index)
	output_df = output_df.reindex(parent_pcs_df.index)
	
	parent_score = calculate_performance(parent_pcs_df.values, output_df.loc[:, 'Predicted'].values.ravel())
	children_score = calculate_performance(children_pcs_df.values, output_df.loc[:, 'Predicted'].values.ravel())

	print("parent score", parent_score)
	print("children score", children_score)
	print("rlipp", parent_score/children_score)

	return parent_score/children_score


def calculate_pcs(hiddens_df, mask=[], variance_explained=1):

	"""Calculates a principal components 
    
	Parameters	
	----------
	term : str 
		Term name
	genes : list of str
		List of gene names that fall within this term 
	variance_explained : float
		(0,1]. The number of principal components to include needed to reach 
		this threhsold.  
	f : callable 
		A function that maps the term name to a file location 
		e.g. f(name) -> "name.txt"
	"""

	# extract only the hidden weights related to the term
	hiddens_df = hiddens_df.loc[mask]

	hiddens = hiddens_df.iloc[:, :-2].values
	scaled_hiddens = scale(hiddens)

	pca = PCA()
	pcs = pca.fit_transform(scaled_hiddens)

	n = len(pcs[0])
	print("number of neurons: %d" % n)

	if variance_explained < 1:
		n = get_number_of_pcs(pca.explained_variance_ratio_, variance_explained)

	if n < 2:
		n = 2

	print("number of PCs: %d" % n)

	pcs_df = pd.DataFrame(pcs[:, :n], index=hiddens_df.index)
	output_df = hiddens_df.iloc[:, -2:]
	output_df.columns = ['Predicted', 'Measured']

	return pcs_df, output_df, mask


# build a mask vector
# True if the cell line has a gene mutation belonging to the term, False otherwise
def check_index(index, geneset, cell2gene, sep='$'):
	"""Subsets the index for genotypes in a geneset"""

	mask = []
	
	for ind in index:
		try:
			cell, _, _ = ind.split(sep)
		except ValueError:
			raise ValueError("Either/both cell or/and drug information is missing")

		cell_geneset = cell2gene[cell]
		overlap = set(geneset).intersection(cell_geneset)

		if len(overlap) > 0:
			mask.append(True)
		else:
			mask.append(False)

	return mask


# run Ridge Regression model using weights
def calculate_performance(features, output, cv=5):
	if features.shape[0] < cv:
		return calculate_performane_noCV(features, output)

	else:
		model = RidgeCV(cv=cv)
		model.fit(features, output)
		predicted = model.predict(features)

		return spearmanr(predicted, output).correlation
		

# run Ridge Regression model without cross validation
def calculate_performane_noCV(features, output):
	model = Ridge()
	model.fit(features, output)
	predicted = model.predict(features)

	return spearmanr(predicted, output).correlation


# calculate the number of PCs 
def get_number_of_pcs(explained_variance_ratio, threshold):
	"""Gets the correct number of PC for a total variance explained"""

	total = 0
	ind = 0
	while total < threshold:
		total += explained_variance_ratio[ind]
		ind += 1

	return ind


# get new column names
def get_new_column_names(df, name='pc'):
	"""Change columns to a different name"""
	
	names = ['%s%d' % (name, ind) for ind in range(len(df.columns))]
	df.columns = names

	return df


# get all possible parent-children pairs
def get_parent_children_tuple(ont):
	tuples = []
	terminals = []
	for term in ont.terms:
		if ont.parent_2_child[term]:
			tuples.append((term, ont.parent_2_child[term]))
		else:
			terminals.append(term)
			
	return tuples, terminals



# main function
def main():
	global input_dir

	opt = parser.parse_args()

	input_dir = opt.hidden
	output_prefix = opt.output

	drug_vec_len = opt.druglen

	# load mapping between cells and genes #################################
	cell2gene = {}
	with open(opt.cell2gene, 'r') as fi:
		for line in fi:
			tokens = line.strip().split('\t')

			if len(tokens) > 1:
				cell2gene[tokens[0]] = tokens[1].split(',')
			else:
				cell2gene[tokens[0]] = []
	########################################################################	


	# load mapping between gene and index ##################################
	gene2id = {}
	with open(opt.gene2id, 'r') as fi:
		for line in fi:
			tokens = line.strip().split('\t')
			gene2id[tokens[1]] = int(tokens[0])
	########################################################################


	# load drug correlation file and get top X drugs #######################
    # and find relevant lines ##############################################
	ind2drug = {}
	drug2ind = {}
	with open(opt.drugcorr, 'r') as fi:
		fi.readline()
		for line in fi:
			tokens = line.strip().split('\t')
			drug2ind[tokens[3]] = tokens[2]
			ind2drug[tokens[2]] = tokens[3]		

	# if there is any tissue specified, another layer of filterin
	i = 0
	drug2line = defaultdict(list)

	if opt.tissue != '': # one specific tissue will be used
		with open(opt.data, 'r') as fi:
			for line in fi:
				tokens = line.strip().split('\t')

				cell = tokens[0]
				drug = tokens[1]

				if drug in drug2ind:
					if opt.tissue.upper() in cell:
						drug2line[drug2ind[drug]].append(i)
				i += 1

	else: # all tissues included
		with open(opt.data, 'r') as fi:
			for line in fi:
				tokens = line.strip().split('\t')
				drug = tokens[1]

				if drug in drug2ind:
					drug2line[drug2ind[drug]].append(i) 

				i += 1
	

	# load ontology file ####################################################
	# getting ontology structures

	df = pd.read_csv(opt.onto, sep='\t', names=['Parent', 'Child', 'Type'])
	
	term_term_rels = df.loc[df['Type'] == 'default']
	term_gene_rels = df.loc[df['Type'] == 'gene']

	ont = Ontology.from_table(term_term_rels, parent='Parent', child='Child', 
								mapping=term_gene_rels, mapping_parent='Parent', mapping_child='Child')

	ont.propagate(direction='forward', inplace=True)
	##########################################################################

	# getting mapping between term and genes
	term2gene = { t: np.array(ont.genes)[g] for t, g in ont.term_2_gene.items() }	

	# getting all possible pairs of parent-children: children are in a list
	parent_children_tuples, terminals = get_parent_children_tuple(ont)

	def rlipp_wrapper(parent, children, linemask):
		score = rlipp(parent, children, term2gene, cell2gene, linemask)
		return parent, score

	'''
	# parallelization does not really work
	with Pool(processes=4) as pool:
		results = pool.starmap(rlipp_wrapper, parent_children_tuples)
	'''

	for drug in drug2line:
		dataind = drug2line[drug]

		print("processing: %s(%s)" % (drug, ind2drug[drug]))

		outputfile = "%s_%s.txt" % (output_prefix, drug)
	
		with open(outputfile, 'w', buffering=0) as fo:

			for t in terminals:
				score = rlipp_single(t, term2gene, cell2gene, gene2id, drug_vec_len, dataind, mask=[], variance_explained=0.9)
				fo.write("%s\tt\t%f\n" % (t, score))
	
			for p, c in parent_children_tuples:
				_, score = rlipp_wrapper(p, c, dataind)
				fo.write("%s\ti\t%f\n" % (p, score))



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='RLIPP calculation')
	parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
	parser.add_argument('-cell2gene', help='Cell to gene mapping file', type=str, default="/data/cellardata/drug_study/DrugCell/input_data/data/bp/cell_mutations_bp.txt")
	parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default="/data/cellardata/drug_study/DrugCell/input_data/data/bp/cell_mutations_bp_gene2id.txt")
	parser.add_argument('-hidden', help='Input directory: where hidden.built files are', type=str)
	parser.add_argument('-output', help='Output file prefix', type=str)
	parser.add_argument('-drugcorr', help='Input file: drug correlation file', type=str)
	parser.add_argument('-data', help='File containing all data points', type=str, default="/data/cellardata/drug_study/DrugCell/input_data/5folds/drugcell_all.txt")
	parser.add_argument('-druglen', help='File containing all data points', type=int)
	parser.add_argument('-tissue', help='Specify a tissue in interest', type=str, default='')

	main()
