import numpy as np
import time
import torch
import torch.nn as nn
from torch._six import inf
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import optuna
from optuna.trial import TrialState

import util
from training_data_wrapper import *
from drugcell_nn import *


class OptunaGradientNNTrainer(GradientNNTrainer):

	def __init__(self, opt):
		super().__init__(opt)


	def exec_study(self):
		study = optuna.create_study(direction=["minimize", "maximize"])
		study.optimize(self.train_model, n_trials=10)
		self.print_result(study)


	def setup_trials(self, trial):

		self.data_wrapper.genotype_hiddens = trial.suggest_categorical("neurons_per_node", [2, 4, 6, 8, 10, 12])
		#self.data_wrapper.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
		self.alpha = trial.suggest_categorical("alpha", [0.1, 0.2, 0.3, 0.4])

		for key, value in trial.params.items():
			print("{}: {}".format(key, value))


	def train_model(self, trial):

		epoch_start_time = time.time()
		model_scores = []
		pareto_front = []

		self.setup_trials(trial)

		train_features, train_labels = self.data_wrapper.train_data
		train_feature, val_feature, train_label, val_label = train_test_split(train_features, train_labels, test_size = 0.1, shuffle = False)

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_label_gpu = Variable(train_label.cuda(self.data_wrapper.cuda))
		val_label_gpu = Variable(val_label.cuda(self.data_wrapper.cuda))
		train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label), batch_size = self.data_wrapper.batchsize, shuffle = False)
		val_loader = du.DataLoader(du.TensorDataset(val_feature, val_label), batch_size = self.data_wrapper.batchsize, shuffle = False)

		optimizer = torch.optim.Adam(self.model.parameters(), lr = self.data_wrapper.learning_rate, betas = (0.9, 0.99), eps = 1e-05)
		optimizer.zero_grad()

		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
			_gradnorms = torch.empty(len(train_loader)).cuda(CUDA_ID) # tensor for accumulating grad norms from each batch in this epoch
			epoch_scores = []

			for i, (inputdata, labels) in enumerate(train_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features, self.data_wrapper.drug_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				aux_out_map,_ = self.model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim = 0)

				total_loss = 0
				for name, output in aux_out_map.items():
					loss = nn.MSELoss()
					if name == 'final':
						total_loss += loss(output, cuda_labels)
					else:
						total_loss += self.alpha * loss(output, cuda_labels)
				total_loss.backward()

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				_gradnorms[i] = util.get_grad_norm(model.parameters()).unsqueeze(0) # Save gradnorm for batch
				optimizer.step()

			gradnorms = sum(_gradnorms).unsqueeze(0).cpu().numpy()[0] # Save total gradnorm for epoch
			train_corr = util.pearson_corr(train_predict, train_label_gpu)

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
					val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim = 0)

			val_corr = util.pearson_corr(val_predict, val_label_gpu)

			epoch_scores.append(gradnorms)
			epoch_scores.append(val_corr)
			model_scores.append(epoch_scores)
			pareto_ids = self.calc_pareto_front(model_scores)
			model_scores = model_scores[pareto_ids]

			epoch_end_time = time.time()
			print("epoch {}\ttrain_corr {:.3f}\tval_corr {:.3f}\ttotal_loss {:.3f}\tgrad_norm {:.3f}\telapsed_time {}".format(epoch, train_corr, val_corr, total_loss, gradnorms, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

		return model_scores[-1]


	def print_result(self, study):

		fig_params = optuna.visualization.plot_param_importances(study)
		fig_params.save(self.data_wrapper.modeldir + "/param_importance.png")

		fig_pareto = optuna.visualization.plot_pareto_front(study)
		fig_pareto.save(self.data_wrapper.modeldir + "/pareto_front.png")


	# Calculate Pareto front for gradient norm and validation correlation
	# scores is an [Nx2] vector where the 1st column is gradient norm
	def calc_pareto_front(self, scores):
		if len(objective_scores) <= 1:
			return objective_scores

		vec_len = scores.shape[0]
		vec_ids = np.arange(vec_len)
		pareto_front = np.ones(vec_len, dtype=bool)
		for i in range(vec_len):
			for j in range(vec_len):
				if i == j:
					continue
				if (scores[j, 0] <= scores[i, 0] and scores[j, 1] > scores[i, 1]) or (scores[j, 0] < scores[i, 0] and scores[j, 1] >= scores[i, 1]):
					pareto_front[i] = 0
					break

		return vec_ids[pareto_front]
