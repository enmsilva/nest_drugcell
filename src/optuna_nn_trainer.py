import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import optuna
from optuna.trial import TrialState

import util
from nn_trainer import *
from training_data_wrapper import *


class OptunaNNTrainer(NNTrainer):

	def __init__(self, opt):
		super().__init__(opt)


	def exec_study(self):
		study = optuna.create_study(direction="maximize")
		study.optimize(self.train_model, n_trials=50)
		self.print_result(study)


	def setup_trials(self, trial):

		self.data_wrapper.genotype_hiddens = trial.suggest_categorical("neurons_per_node", [2, 4, 6, 8])
		self.data_wrapper.lr = trial.suggest_float("learning_rate", 1e-6, 5e-1, log=True)
		self.data_wrapper.wd = trial.suggest_float("weight_decay", 1e-6, 5e-1, log=True)
		self.data_wrapper.alpha = trial.suggest_categorical("alpha", [0.2, 0.5, 0.8, 1.0])

		for key, value in trial.params.items():
			print("{}: {}".format(key, value))


	def train_model(self, trial):

		epoch_start_time = time.time()
		max_corr = 0

		self.setup_trials(trial)

		train_feature, train_label, val_feature, val_label = self.data_wrapper.prepare_train_data()

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_label_gpu = Variable(train_label.cuda(self.data_wrapper.cuda))
		val_label_gpu = Variable(val_label.cuda(self.data_wrapper.cuda))
		train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label), batch_size=self.data_wrapper.batchsize, shuffle=True)
		val_loader = du.DataLoader(du.TensorDataset(val_feature, val_label), batch_size=self.data_wrapper.batchsize, shuffle=True)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.wd)
		optimizer.zero_grad()

		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)

			for i, (inputdata, labels) in enumerate(train_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features, self.data_wrapper.drug_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				# Here term_NN_out_map is a dictionary
				aux_out_map, _ = self.model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

				total_loss = 0
				for name, output in aux_out_map.items():
					loss = nn.MSELoss()
					if name == 'final':
						total_loss += loss(output, cuda_labels)
					else:
						total_loss += self.data_wrapper.alpha * loss(output, cuda_labels)
				total_loss.backward()

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				optimizer.step()

			train_corr = util.get_drug_corr_median(train_predict, train_label_gpu, train_feature)

			self.model.eval()

			val_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)

			for i, (inputdata, labels) in enumerate(val_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features, self.data_wrapper.drug_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				aux_out_map, _ = self.model(cuda_features)

				if val_predict.size()[0] == 0:
					val_predict = aux_out_map['final'].data
				else:
					val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)

			val_corr = util.get_drug_corr_median(val_predict, val_label_gpu, val_feature)

			if val_corr >= max_corr:
				max_corr = val_corr

			trial.report(val_corr, epoch)

			epoch_end_time = time.time()
			print("epoch {}\ttrain_corr {:.5f}\tval_corr {:.5f}\ttotal_loss {:.3f}\telapsed_time {}".format(epoch, train_corr, val_corr, total_loss, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

		return max_corr


	def print_result(self, study):

		pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
		complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

		print("Study statistics:")
		print("Number of finished trials:", len(study.trials))
		print("Number of pruned trials:", len(pruned_trials))
		print("Number of complete trials:", len(complete_trials))

		print("Best trial:")
		trial = study.best_trial

		print("Value: ", trial.value)

		print("Params:")
		for key, value in trial.params.items():
			print("{}: {}".format(key, value))

		fig_params = optuna.visualization.plot_param_importances(study)
		fig_params.save(self.data_wrapper.modeldir + "/param_importance.png")
