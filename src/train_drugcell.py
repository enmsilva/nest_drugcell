import argparse

from nn_trainer import *
from optuna_nn_trainer import *
from gradient_nn_trainer import *
from optuna_gradient_nn_trainer import *


def main():

	torch.set_printoptions(precision = 5)

	parser = argparse.ArgumentParser(description = 'Train DrugCell')
	parser.add_argument('-onto', help = 'Ontology file used to guide the neural network', type = str)
	parser.add_argument('-train', help = 'Training dataset', type = str)
	parser.add_argument('-val', help = 'Validation dataset', type = str, default = "")
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

	if opt.optimize == 0:
		GradientNNTrainer(opt).train_model()
	else:
		OptunaGradientNNTrainer(opt).exec_study()


if __name__ == "__main__":
	main()
