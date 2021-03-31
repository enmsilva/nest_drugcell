import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import optuna
from optuna.trial import TrialState

import util
from drugcell_NN import *
from training_data_wrapper import *

CUDA_ID = 0


def train_model(data_wrapper, model, train_feature, train_label, val_feature, val_label):

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
		print("epoch %d\ttrain_corr %.4f\tval_corr %.4f\ttotal_loss %.4f\telapsed_time %s" % (epoch, train_corr, val_corr, total_loss, epoch_end_time - epoch_start_time))
		epoch_start_time = epoch_end_time

		if val_corr >= max_corr:
			max_corr = val_corr

	return max_corr


def train_model(trial, data_wrapper, model, train_feature, train_label, val_feature, val_label):

	epoch_start_time = time.time()
	max_corr = 0
	max_corr_count = 0

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

	# Generate the optimizers.
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
	optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=data_wrapper.learning_rate)
	optimizer.zero_grad()

	print("Learning rate = %d\tNeurons = %d\tOptimizer = %s" %(data_wrapper.learning_rate, data_wrapper.num_hiddens_genotype, optimizer_name))

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
		if val_corr is None:
			val_corr = 0

		epoch_end_time = time.time()
		print("epoch %d\ttrain_corr %.4f\tval_corr %.4f\ttotal_loss %.4f\telapsed_time %s" % (epoch, train_corr, val_corr, total_loss, epoch_end_time - epoch_start_time))
		epoch_start_time = epoch_end_time

		if val_corr >= max_corr:
			max_corr = val_corr
		else:
			max_corr_count += 1

		trial.report(val_corr, epoch)

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

		if max_corr_count >= 10:
			break

	return max_corr


def exec_trial_training(trial, opt):

	opt.genotype_hiddens = trial.suggest_int("neurons_per_node", 1, 12)
	opt.lr = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
	data_wrapper = TrainingDataWrapper(opt)

	# Create the neural network
	model = drugcell_nn(data_wrapper)
	model.cuda(CUDA_ID)

	train_features, train_labels = data_wrapper.train_data
	train_feat, val_feat, train_label, val_label = train_test_split(train_features, train_labels, test_size = 0.1, shuffle = False)

	train_model(trial, data_wrapper, model, train_feat, train_label, val_feat, val_label)
	torch.save(model, data_wrapper.modeldir + '/model_final.pt')


def exec_training(opt):

	data_wrapper = TrainingDataWrapper(opt)

	# Create the neural network
	model = drugcell_nn(data_wrapper)
	model.cuda(CUDA_ID)

	train_features, train_labels = data_wrapper.train_data
	train_feat, val_feat, train_label, val_label = train_test_split(train_features, train_labels, test_size = 0.1, shuffle = False)

	train_model(data_wrapper, model, train_feat, train_label, val_feat, val_label)
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
	parser.add_argument('-optimize', help = 'Hyper-parameter optimization', type = int, default = 0)

	opt = parser.parse_args()
	CUDA_ID = opt.cuda

	if opt.optimize == 0:
		exec_training(opt)
	else:

		study = optuna.create_study(direction="maximize")

		study.optimize(lambda trial: exec_trial_training(trial, opt), n_trials=200)

		pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
		complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

		print("Study statistics: ")
		print("  Number of finished trials: ", len(study.trials))
		print("  Number of pruned trials: ", len(pruned_trials))
		print("  Number of complete trials: ", len(complete_trials))

		print("Best trial:")
		trial = study.best_trial

		print("  Value: ", trial.value)

		print("  Params: ")
		for key, value in trial.params.items():
			print("    {}: {}".format(key, value))


if __name__ == "__main__":
	main()
