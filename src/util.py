import torch
from torch._six import inf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import scale

def pearson_corr(x, y):
	xx = x - torch.mean(x)
	yy = y - torch.mean(y)

	return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))


def standardize_train_data(train_df, zscore_method, std_file):

	std_file_out = open(std_file, "w")

	if zscore_method == 'zscore':
		train_df[5] = df.groupby([3,4])[2].transform(lambda x: scale(x))
		for name, group in train_df.groupby([3,4])[2]:
			std_file_out.write("{}\t{}\t{}\t{}\n".format(name[0], name[1], group.mean(), group.std()))

	elif zscore_method == 'robustz':
		train_df[5] = train_df.groupby([3,4])[2].transform(lambda x: robust_scale(x))
		iqr = group.quantile(0.75) - group.quantile(0.25)
		for name, group in train_df.groupby([3,4])[2]:
			std_file_out.write("{}\t{}\t{}\t{}\n".format(name[0], name[1], group.median(), iqr))
	else:
		train_df[5] = train_df[2]
		for name, group in train_df.groupby([3,4])[2]:
			std_file_out.write("{}\t{}\t{}\t{}\n".format(name[0], name[1], 0, 1))

	std_file_out.close()

	train_df.drop([2,3,4], axis=1, inplace=True)
	train_df.columns = range(train_df.shape[1])
	return train_df


def load_train_data(file_name, cell2id, drug2id, zscore_method, std_file):
	feature = []
	label = []

	train_df = pd.read_csv(file_name, sep='\t', header=None)
	train_df = standardize_train_data(train_df, zscore_method, std_file)

	for row in train_df.values:
		feature.append([cell2id[row[0]], drug2id[row[1]]])
		label.append([float(row[2])])

	return feature, label


def prepare_train_data(train_file, val_file, cell2id_mapping, drug2id_mapping, zscore_method, std_file):

	train_features, train_labels = load_train_data(train_file, cell2id_mapping, drug2id_mapping, zscore_method, std_file)

	val_features, val_labels = load_pred_data(val_file, cell2id_mapping, drug2id_mapping, zscore_method, std_file)

	return (torch.Tensor(train_features), torch.FloatTensor(train_labels), torch.Tensor(val_features), torch.FloatTensor(val_labels))


def standardize_test_data(test_df, zscore_method, std_file):

	std_file_df = pd.read_csv(std_file, sep='\t', header=None)

	merged = pd.merge(test_df, std_file_df, how="left", left_on=[3, 4], right_on=[0, 1], sort=False)
	merged = merged[["0_x", "1_x", "2_x", "2_y", "3_y"]]
	merged.columns = range(merged.shape[1])

	n = merged.shape[1]
	merged[n] = (merged[2] - merged[n-2]) / merged[n-1]
	merged = merged[[0, 1, n-1]]
	merged.columns = range(merged.shape[1])
	return merged


def load_pred_data(file_name, cell2id, drug2id, zscore_method, std_file):
	feature = []
	label = []

	test_df = pd.read_csv(file_name, sep='\t', header=None)
	test_df = standardize_test_data(test_df, zscore_method, std_file)

	for row in test_df.values:
		feature.append([cell2id[row[0]], drug2id[row[1]]])
		label.append([float(row[2])])

	return feature, label


def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file, zscore_method, std_file):

	# load mapping files
	cell2id_mapping = load_mapping(cell2id_mapping_file, 'cell lines')
	drug2id_mapping = load_mapping(drug2id_mapping_file, 'drugs')

	test_features, test_labels = load_pred_data(test_file, cell2id_mapping, drug2id_mapping, zscore_method, std_file)

	return (torch.Tensor(test_features), torch.Tensor(test_labels)), cell2id_mapping, drug2id_mapping


def load_mapping(mapping_file, mapping_type):

	mapping = {}

	file_handle = open(mapping_file)

	for line in file_handle:
		line = line.rstrip().split()
		mapping[line[1]] = int(line[0])

	file_handle.close()
	print('Total number of {} = {}'.format(mapping_type, len(mapping)))

	return mapping


def build_input_vector(input_data, cell_features, drug_features):
	genedim = len(cell_features[0,:])
	drugdim = len(drug_features[0,:])
	feature = np.zeros((input_data.size()[0], (genedim+drugdim)))

	for i in range(input_data.size()[0]):
		feature[i] = np.concatenate((cell_features[int(input_data[i,0])], drug_features[int(input_data[i,1])]), axis=None)

	feature = torch.from_numpy(feature).float()
	return feature


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim, cuda_id):
	term_mask_map = {}
	for term, gene_set in term_direct_gene_map.items():
		mask = torch.zeros(len(gene_set), gene_dim).cuda(cuda_id)
		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1
		term_mask_map[term] = mask
	return term_mask_map


# Update variance for every weight using Welford's online algorithm
def update_variance(welford_set, new_weight):
	(n, mean, M2) = welford_set
	n += 1
	delta = abs(new_weight - mean)
	mean += delta/n
	delta2 = abs(new_weight - mean)
	M2 += delta * delta2
	return (n, mean, M2)


def get_grad_norm(model_params, norm_type):
	"""Gets gradient norm of an iterable of model_params.
	The norm is computed over all gradients together, as if they were
	concatenated into a single vector. Gradients are modified in-place.
	Arguments:
		model_params (Iterable[Tensor] or Tensor): an iterable of Tensors or a
			single Tensor that will have gradients normalized
		norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
			infinity norm.
	Returns:Total norm of the model_params (viewed as a single vector).
	"""
	if isinstance(model_params, torch.Tensor): # check if parameters are tensorobject
		model_params = [model_params] # change to list
	model_params = [p for p in model_params if p.grad is not None] # get list of params with grads
	norm_type = float(norm_type) # make sure norm_type is of type float
	if len(model_params) == 0: # if no params provided, return tensor of 0
		return torch.tensor(0.)

	device = model_params[0].grad.device # get device
	if norm_type == inf: # infinity norm
		total_norm = max(p.grad.detach().abs().max().to(device) for p in model_params)
	else: # total norm
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in model_params]), norm_type)
	return total_norm
