import torch
import numpy as np

def pearson_corr(x, y):
	xx = x - torch.mean(x)
	yy = y - torch.mean(y)

	return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))


def load_train_data(file_name, cell2id, drug2id):
	feature = []
	label = []

	with open(file_name, 'r') as fi:
		for line in fi:
			tokens = line.strip().split('\t')

			feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
			label.append([float(tokens[2])])

	return feature, label


def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):

	# load mapping files
	cell2id_mapping = load_mapping(cell2id_mapping_file)
	drug2id_mapping = load_mapping(drug2id_mapping_file)

	test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

	print('Total number of cell lines = %d' % len(cell2id_mapping))
	print('Total number of drugs = %d' % len(drug2id_mapping))

	return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping


def load_mapping(mapping_file):

	mapping = {}

	file_handle = open(mapping_file)

	for line in file_handle:
		line = line.rstrip().split()
		mapping[line[1]] = int(line[0])

	file_handle.close()

	return mapping


def prepare_train_data(train_file, cell2id_mapping, drug2id_mapping):

	train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)

	print('Total number of cell lines = %d' % len(cell2id_mapping))
	print('Total number of drugs = %d' % len(drug2id_mapping))

	return (torch.Tensor(train_feature), torch.FloatTensor(train_label))


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
