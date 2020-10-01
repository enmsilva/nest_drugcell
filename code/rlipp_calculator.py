import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from scipy import stats

class RLIPPCalculator():
    
    def __init__(self, args):
        self.ontology = pd.read_csv(args.ontology, sep='\t', header=None, names=['S', 'T', 'I'])
        self.test_df = pd.read_csv(args.test, sep='\t', header=None, names=['C', 'D', 'AUC'])
        self.predicted_vals = np.loadtxt(args.predicted)
        self.drugs = pd.read_csv(args.drug_index, sep='\t', header=None, names=['I', 'D'])['D']
        self.genes = pd.read_csv(args.gene_index, sep='\t', header=None, names=['I', 'G'])['G']
        self.cell_index = pd.read_csv(args.cell_index, sep="\t", header=None, names=['I', 'C'])
        self.cell_mutation = np.loadtxt(args.cell_mutation, delimiter=',')
        self.out_file = args.output
        
        self.hidden_dir = args.hidden
        if not self.hidden_dir.endswith('/'):
            self.hidden_dir += '/'
        
        self.drug_count = args.drug_count
        if self.drug_count == 0:
            self.drug_count = len(drugs)
        
        self.terms = self.ontology['S'].unique().tolist()
        
        #self.create_gene_hidden_files()


    def create_gene_hidden_files(self):
        cell_id_map = dict(zip(self.cell_index['C'], self.cell_index['I']))
        cell_line_ids = np.array([cell_id_map[x] for x in self.test_df['C'].tolist()])
        for i, gene in enumerate(self.genes):
            file_name = self.hidden_dir + gene + '.hidden'
            mat_data_sub = self.cell_mutation[cell_line_ids, i].ravel()
            np.savetxt(file_name, mat_data_sub, fmt='%.3f')

    
    def get_drug_pos_map(self):
        drug_pos_map = {d:[] for d in self.drugs}
        for i, row in self.test_df.iterrows():
            drug_pos_map[row['D']].append(i)
        return drug_pos_map


    def sort_drugs_corr(self, drug_pos_map):
        drug_corr_map = {}
        for d in self.drugs:
            test_vals = np.take(np.array(self.test_df['AUC']), drug_pos_map[d])
            pred_vals = np.take(self.predicted_vals, drug_pos_map[d])
            drug_corr_map[d] = stats.spearmanr(test_vals, pred_vals)[0]
        return {drug:corr for drug,corr in sorted(drug_corr_map.items(), key=lambda item:item[1], reverse=True)}

    
    def load_feature(self, term, size):
        file_name = self.hidden_dir + term + '.hidden'
        return np.loadtxt(file_name, usecols=range(size))


    def load_all_features(self):
        feature_map = {}
        for t in self.terms:
            feature_map[t] = load_feature(t, 6)
        for g in self.genes:
            feature_map[g] = load_feature(g, 1)
        return feature_map


    def get_child_features(self, feature_map, term, index_list):
        child_features = []
        children = [row['T'] for _,row in self.ontology.iterrows() if row['S']==term]
        for child in children:
            child_features.append(np.take(feature_map[child], index_list))
        return np.column_stack((f for f in child_features))


    def exec_lm(self, X, y):
        regr = RidgeCV(fit_intercept=False, cv=5)
        regr.fit(X, y)
        y_pred = regr.predict(X)
        return stats.spearmanr(y_pred, y)[0]


    def calc_scores(self):
        print('Starting score calculation')
        outf = open(self.out_file, "w")
        outf.write('Drug\tTerm\tP_rho\tC_rho\tRLIPP\n')
        
        drug_pos_map = self.get_drug_pos_map()
        sorted_drugs = self.sort_drugs_corr(drug_pos_map).keys()
        
        feature_map = self.load_all_features()
        print('feature map created')
        for i,d in enumerate(sorted_drugs):
            if i == self.drug_count:
                break
            y = np.take(self.predicted_vals, drug_pos_map[d])
            for t in self.terms:
                X_parent = np.take(feature_map[t], drug_pos_map[d])
                X_child = self.get_child_features(feature_map, t, drug_pos_map[d])
                p_rho = self.exec_lm(X_parent, y)
                c_rho = self.exec_lm(X_child, y)
                rlipp = (p_rho - c_rho)/c_rho
                result = '{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(d, t, p_rho, c_rho, rlipp)
                outf.write(result)
            print('Drug ' + str(i+1) + ' complete!')
        outf.close()

