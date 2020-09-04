import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy import stats

HOMEDIR = ''


def load_features(term, size):
	file_name = HOMEDIR + '/rlipp/hidden/' + term + '.hidden'
	return np.loadtxt(file_name, usecols=range(size))

def get_features(term, is_term, index_list):
	features = []
	if term in terms:
		features = np.loadtxt(file_name, usecols=range(6))
	else:
		features = np.loadtxt(file_name, usecols=range(1))
	return np.take(features, index_list, axis=0)


def get_child_features(term, index_list, ontology):
	child_features = []
	children = [row['T'] for _,row in ontology.iterrows() if row['S']==term]
	for child in children:
		child_features.append(get_features(child, index_list))
	return np.column_stack((f for f in child_features))


def exec_lm(X, y):
	regr = RidgeCV(fit_intercept=False, cv=5)
	regr.fit(X, y)
	y_pred = regr.predict(X)
	return stats.spearmanr(y_pred, y)[0]


def get_drug_pos_map(test_df, drugs):
	drug_pos_map = {d:[] for d in drugs}
	for i, row in test_df.iterrows():
		drug_pos_map[row['D']].append(i)
	return drug_pos_map


def sort_drugs_corr(drugs, test_df, predicted):
	drug_corr_map = {}
	drug_pos_map = get_drug_pos_map(test_df, drugs)
	for d in drugs:
		test_vals = np.take(np.array(test_df['AUC']), drug_pos_map[d])
		pred_vals = np.take(predicted, drug_pos_map[d])
		drug_corr_map[d] = stats.spearmanr(test_vals, pred_vals)[0]
	return {drug:corr for drug,corr in sorted(drug_corr_map.items(), key=lambda item:item[1], reverse=True)}


def calc_rlipp_scores(drugs, test_df, predicted, ontology, out_file):

	f = open(out_file, "w")
	f.write('Drug\tTerm\tP_rho\tC_rho\tRLIPP\n')
	drug_pos_map = get_drug_pos_map(test_df, drugs)
	sorted_drugs = sort_drugs_corr(drugs, test_df, predicted).keys()
	terms = ontology['S'].unique().tolist()
    
    for t in terms:
		

	for d in sorted_drugs:
		y = np.take(predicted_vals, drug_pos_map[d])
		for t in terms:
			X_parent = get_features(t, drug_pos_map[d])
			X_child = get_child_features(t, drug_pos_map[d], ontology)
			p_rho = exec_lm(X_parent, y)
			c_rho = exec_lm(X_child, y)
			rlipp = (p_rho - c_rho)/c_rho
			result = '{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(d, t, p_rho, c_rho, rlipp)
			f.write(result)
	f.close()


def main():

	HOMEDIR = sys.argv[1]
	test_df = pd.read_csv(HOMEDIR + '/data/drugcell_all.txt', sep='\t', header=None, names=['C', 'D', 'AUC'])
	drugs = pd.read_csv(HOMEDIR + '/data/drug2ind.txt', sep='\t', header=None, names=['I', 'D'])['D']
	ontology = pd.read_csv(HOMEDIR + '/data/drugcell_ont.txt', sep='\t', header=None, names=['S', 'T', 'I'])
	predicted_vals = np.loadtxt(HOMEDIR + '/rlipp/drugcell_all.predict')
	
	out_file = HOMEDIR + '/rlipp/rlipp.out'
	calc_rlipp_scores(drugs, test_df, predicted_vals, ontology, out_file)


if __name__ == "__main__":
	main()
