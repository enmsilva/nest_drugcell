import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import util
from training_data_wrapper import *
from drugcell_nn import *


class NNTrainer():

	def __init__(self, opt):

		self.data_wrapper = TrainingDataWrapper(opt)

		self.model = DrugCellNN(self.data_wrapper)
		self.model.cuda(self.data_wrapper.cuda)


	def train_model(self):

		epoch_start_time = time.time()
		best_model = 0
		max_corr = 0

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

		self.term_feature_variance_map = {} # Map of term -> list of variance of every gene part of that term

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
						total_loss += 0.3 * loss(output, cuda_labels)
				total_loss.backward()

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				optimizer.step()

				self.update_variance(term_mask_map)

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

			if val_corr >= max_corr:
				max_corr = val_corr

			epoch_end_time = time.time()
			print("epoch %d\ttrain_corr %.4f\tval_corr %.4f\ttotal_loss %.4f\telapsed_time %s" % (epoch, train_corr, val_corr, total_loss, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

		self.finalize_variance()
		mean_variance_map, mean_viann_score_map = self.calc_feature_importance(term_mask_map)
		mutations_per_gene = np.count_nonzero(self.data_wrapper.cell_features.transpose() == 1, axis=1)
		for gene, score in viann_score_map.items():
			mut_freq = mutations_per_gene[self.data_wrapper.gene_id_mapping[gene]]/len(mutations_per_gene)
			print("Gene %s\t Mutation_frequency %.4f\t Variance %.4f\t VIANN_score %.4f" % (gene, mut_freq, mean_variance_map[gene], score))

		torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')

		return max_corr


	# Update Variance of every gene per term using Welford's online method
	def update_variance(self, term_mask_map):
		model_weights_map = self.model.get_model_weights(term_mask_map, '_direct_gene_layer.weight')
		for term, term_weights in model_weights_map.items():
			feature_welford_set_list = [(0, 0.0, 0.0)] * len(term_weights)
			for i, weight in enumerate(term_weights):
				if term in self.term_feature_variance_map:
					feature_welford_set_list = self.term_feature_variance_map[term]
				feature_welford_set_list[i] = util.update_variance(feature_welford_set_list[i], weight)
			self.term_feature_variance_map[term] = feature_welford_set_list


	# Calculate variance for every gene per term
	def finalize_variance(self):
		for term, feature_welford_set_list in self.term_feature_variance_map.items():
			feature_variance_list = torch.zeros(len(feature_welford_set_list)).cuda(self.data_wrapper.cuda)
			for i, elem in enumerate(feature_welford_set_list):
				(n, mean, M2) = elem
				feature_variance_list[i] = M2/n
			self.term_feature_variance_map[term] = feature_variance_list


	# Calculate VIANN score for every gene; VIANN score = sum(Variance x final weight)
	def calc_feature_importance(self, term_mask_map):

		viann_scores = torch.zeros(self.model.gene_dim).cuda(self.data_wrapper.cuda) # List of VIANN scores of every gene; each entry contains the sum of weighted variance
		viann_freq = torch.zeros(self.model.gene_dim).cuda(self.data_wrapper.cuda)   # No of VIANN scores of every gene = No of subsystems containing that gene
		variance_sum = torch.zeros(self.model.gene_dim).cuda(self.data_wrapper.cuda) # List of sum of variance of every gene
		final_weights_map = self.model.get_model_weights(term_mask_map, '_direct_gene_layer.weight')
		for term, final_term_weights in final_weights_map.items():
			feature_variance_list = self.term_feature_variance_map[term]
			weighted_variance_list = torch.mul(final_term_weights, feature_variance_list)
			for i, gene_id in enumerate(self.model.term_direct_gene_map[term]):
				variance_sum[gene_id] += feature_variance_list[i]
				viann_scores[gene_id] += weighted_variance_list[i]
				viann_freq[gene_id] += 1

		mean_viann_score_map = {}
		mean_variance_map = {}
		for i, gene in enumerate(self.data_wrapper.gene_id_mapping.keys()):
			mean_variance_map[gene] = variance_sum[i].item()/viann_freq[i].item()
			mean_viann_score_map[gene] = viann_scores[i].item()/viann_freq[i].item()

		mean_variance_map = {g:v for g,v in sorted(mean_variance_map.items(), key=lambda item:item[1], reverse=True)}
		mean_viann_score_map = {g:sc for g,sc in sorted(mean_viann_score_map.items(), key=lambda item:item[1], reverse=True)}

		return mean_variance_map, mean_viann_score_map
