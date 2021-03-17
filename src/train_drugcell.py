import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as du
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import util
from drugcell_NN import *
from training_data_wrapper import *

CUDA_ID = 0


def train_model(data_wrapper, model, train_feature, train_label, val_feature, val_label, fold):

	epoch_start_time = time.time()
	best_model = 0
	max_corr = 0

	term_mask_map = util.create_term_mask(model.term_direct_gene_map, model.gene_dim, CUDA_ID)
	for name, param in model.named_parameters():
		term_name = name.split('_')[0]
		if '_direct_gene_layer.weight' in name:
			param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
		else:
			param.data = param.data * 0.1

	train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
	val_label_gpu = torch.autograd.Variable(val_label.cuda(CUDA_ID))
	train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label), batch_size = data_wrapper.batchsize, shuffle = False)
	val_loader = du.DataLoader(du.TensorDataset(val_feature, val_label), batch_size = data_wrapper.batchsize, shuffle = False)

	optimizer = torch.optim.Adam(model.parameters(), lr = data_wrapper.learning_rate, betas = (0.9, 0.99), eps = 1e-05)
	optimizer.zero_grad()

	for epoch in range(data_wrapper.epochs):

		# Train
		model.train()
		train_predict = torch.zeros(0, 0).cuda(CUDA_ID)

		for i, (inputdata, labels) in enumerate(train_loader):
			# Convert torch tensor to Variable
			features = util.build_input_vector(inputdata, data_wrapper.cell_features, data_wrapper.drug_features)

			cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))
			cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

			# Forward + Backward + Optimize
			optimizer.zero_grad()  # zero the gradient buffer

			# Here term_NN_out_map is a dictionary
			aux_out_map, _ = model(cuda_features)

			if train_predict.size()[0] == 0:
				train_predict = aux_out_map['final'].data
			else:
				train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim = 0)

			total_loss = 0
			for name, output in aux_out_map.items():
				loss = nn.MSELoss()
				total_loss += loss(output, cuda_labels)

			total_loss.backward()

			for name, param in model.named_parameters():
				if '_direct_gene_layer.weight' not in name:
					continue
				term_name = name.split('_')[0]
				# print name, param.grad.data.size(), term_mask_map[term_name].size()
				param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

			optimizer.step()

		train_corr = util.pearson_corr(train_predict, train_label_gpu)

		# if epoch % 10 == 0:
		#torch.save(model, data_wrapper.modeldir + '/model_' + str(fold) + '_' + str(epoch) + '.pt')

		# Test: random variables in training mode become static
		model.eval()

		val_predict = torch.zeros(0, 0).cuda(CUDA_ID)

		for i, (inputdata, labels) in enumerate(val_loader):
			# Convert torch tensor to Variable
			features = build_input_vector(inputdata, data_wrapper.cell_features, data_wrapper.drug_features)
			cuda_features = Variable(features.cuda(CUDA_ID))

			aux_out_map, _ = model(cuda_features)

			if val_predict.size()[0] == 0:
				val_predict = aux_out_map['final'].data
			else:
				val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim = 0)

		val_corr = util.pearson_corr(val_predict, val_label_gpu)

		epoch_end_time = time.time()
		print("fold %d\tepoch %d\ttrain_corr %.4f\tval_corr %.4f\ttotal_loss %.4f\telapsed_time %s" % (fold, epoch, train_corr, val_corr, total_loss, epoch_end_time - epoch_start_time))
		epoch_start_time = epoch_end_time

		if val_corr >= max_corr:
			max_corr = val_corr
			
	return max_corr


def cross_validate(data_wrapper):

	max_corr = 0
	kfold = KFold(n_splits = 5)

	train_features, train_labels = data_wrapper.train_data

	# Create the neural network
	model = drugcell_nn(data_wrapper)
	model.cuda(CUDA_ID)

	for fold, (train_index, val_index) in enumerate(kfold.split(train_features, train_labels)):
		train_feature_fold = train_features[train_index]
		train_label_fold = train_labels[train_index]
		val_feature_fold = train_features[val_index]
		val_label_fold = train_labels[val_index]
		
		max_corr_fold = train_model(data_wrapper, model, train_feature_fold, train_label_fold, val_feature_fold, val_label_fold, fold)
		if max_corr_fold >= max_corr:
			torch.save(model, data_wrapper.modeldir + '/model_final.pt')
			max_corr = max_corr_fold


def exec_training(data_wrapper):

	train_features, train_labels = data_wrapper.train_data
	train_feat, val_feat, train_label, val_label = train_test_split(train_features, train_labels, test_size = 0.1, shuffle = False)

	# Create the neural network
	model = drugcell_nn(data_wrapper)
	model.cuda(CUDA_ID)
		
	train_model(data_wrapper, model, train_feat, train_label, val_feat, val_label, -1)
	torch.save(model, data_wrapper.modeldir + '/model_final.pt')
			

def main():

	torch.set_printoptions(precision = 5)

	parser = argparse.ArgumentParser(description = 'Train DrugCell')
	parser.add_argument('-onto', help = 'Ontology file used to guide the neural network', type = str)
	parser.add_argument('-train', help = 'Training dataset', type = str)
	parser.add_argument('-epoch', help = 'Training epochs for training', type = int, default = 300)
	parser.add_argument('-lr', help = 'Learning rate', type = float, default = 0.001)
	parser.add_argument('-batchsize', help = 'Batchsize', type = int, default = 5000)
	parser.add_argument('-modeldir', help = 'Folder for trained models', type = str, default = 'MODEL/')
	parser.add_argument('-cuda', help = 'Specify GPU', type = int, default = 0)
	parser.add_argument('-gene2id', help = 'Gene to ID mapping file', type = str)
	parser.add_argument('-drug2id', help = 'Drug to ID mapping file', type = str)
	parser.add_argument('-cell2id', help = 'Cell to ID mapping file', type = str)
	parser.add_argument('-genotype_hiddens', help = 'Mapping for the number of neurons in each term in genotype parts', type = int, default = 6)
	parser.add_argument('-drug_hiddens', help = 'Mapping for the number of neurons in each layer', type = str, default = '100,50,6')
	parser.add_argument('-final_hiddens', help = 'The number of neurons in the top layer', type = int, default = 6)
	parser.add_argument('-genotype', help = 'Mutation information for cell lines', type = str)
	parser.add_argument('-fingerprint', help = 'Morgan fingerprint representation for drugs', type = str)
	parser.add_argument('-cross_val', help = 'Run Cross Validation', type = int, default = 0)

	opt = parser.parse_args()
	training_data_wrapper = TrainingDataWrapper(opt)
	CUDA_ID = opt.cuda

	if opt.cross_val == 0:
		exec_training(training_data_wrapper)
	else:
		cross_validate(training_data_wrapper)


if __name__ == "__main__":
	main()
