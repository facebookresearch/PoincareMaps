# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from poincare_maps import *

import scanpy.api as sc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import warnings
# warnings.filterwarnings("ignore")
import argparse

from sklearn.metrics.pairwise import pairwise_distances
import torch

from model import *
from rsgd import RiemannianSGD
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import timeit

from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter

import umap

from coldict import *

def plot_embedding(x, labels=None, labels_name='labels', labels_order=None, 
					   file_name=None, coldict=None,
					   d1=6.5, d2=6.0, fs=4, ms=20,
					   col_palette=plt.get_cmap("tab10"), bbox=(1.3, 0.7)):    

	idx = np.random.permutation(len(x))
	df = pd.DataFrame(x[idx, :], columns=['x1', 'x2'])
	
	fig = plt.figure(figsize=(d1, d2))
	ax = plt.gca()

	if not (labels is None):
		df[labels_name] = labels[idx]
		if labels_order is None:
			labels_order = np.unique(labels)        
		if coldict is None:
			coldict = dict(zip(labels_order, col_palette[:len(labels)]))
		sns.scatterplot(x="x1", y="x2", hue=labels_name, 
						hue_order=labels_order,
						palette=coldict,
						alpha=1.0, edgecolor="none",
						data=df, ax=ax, s=ms)
		ax.legend(fontsize=fs, loc='outside', bbox_to_anchor=bbox)
			
	else:
		sns.scatterplot(x="x1", y="x2",
						data=df, ax=ax2, s=ms)
	fig.tight_layout()
	ax.axis('off')
	# ax.axis('equal') 

	if file_name:
		plt.savefig(file_name + '.png', format='png', dpi=300)


class Encoder(torch.nn.Module):
	def __init__(self,n_inputs, n_outputs, factor=6, bn='before'):
		super(Encoder, self).__init__()
		n_hidden = factor*128
		if bn == 'before':
			self.net = torch.nn.Sequential(
							torch.nn.Linear(n_inputs, n_hidden),
							torch.nn.BatchNorm1d(n_hidden),
							torch.nn.ReLU(),
							torch.nn.Linear(n_hidden, n_hidden),
							torch.nn.BatchNorm1d(n_hidden),
							torch.nn.ReLU(),
							torch.nn.Linear(n_hidden, n_hidden),
							torch.nn.BatchNorm1d(n_hidden),
							torch.nn.ReLU(),
							torch.nn.Linear(n_hidden, n_hidden),
							torch.nn.BatchNorm1d(n_hidden),
							torch.nn.ReLU(),
							torch.nn.Linear(n_hidden, n_outputs))		
		elif bn == 'after':
			self.net = torch.nn.Sequential(
							torch.nn.Linear(n_inputs, n_hidden),							
							torch.nn.ReLU(),
							torch.nn.BatchNorm1d(n_hidden),
							torch.nn.Linear(n_hidden, n_hidden),							
							torch.nn.ReLU(),
							torch.nn.BatchNorm1d(n_hidden),
							torch.nn.Linear(n_hidden, n_hidden),							
							torch.nn.ReLU(),
							torch.nn.BatchNorm1d(n_hidden),
							torch.nn.Linear(n_hidden, n_hidden),							
							torch.nn.ReLU(),
							torch.nn.BatchNorm1d(n_hidden),
							torch.nn.Linear(n_hidden, n_outputs))		
		else:
			self.net = torch.nn.Sequential(
							torch.nn.Linear(n_inputs, n_hidden),
							torch.nn.ReLU(),
							torch.nn.Linear(n_hidden, n_hidden),
							torch.nn.ReLU(),
							torch.nn.Linear(n_hidden, n_hidden),
							torch.nn.ReLU(),
							torch.nn.Linear(n_hidden, n_hidden),
							torch.nn.ReLU(),
							torch.nn.Linear(n_hidden, n_outputs))		


	def forward(self, x):
		return self.net(x)



def poincare_distance(x):
	# print('computing poincare distance...')
	eps = 1e-5
	boundary = 1 - eps
	
	nx = x.size(0)
	x = x.contiguous()
	x = x.view(nx, -1)
	
	norm_x = th.sum(x ** 2, 1, keepdim=True)
	sqdist = euclidean_distance(x) * 2    
	squnorm = 1 - th.clamp(norm_x, 0, boundary)

	x = (sqdist / th.mm(squnorm, squnorm.t())) + 1
	z = th.sqrt(th.pow(x, 2) - 1)
	
	return th.log(x + z)


def train_model(coordinates_from, coordinates_to, file_name, method_type, n_epochs=1000, 
	space='poincare', lr=1e-4, wd=1e-3, batch_size=8, n_warmup=3000, cuda=0, tb=0, bn='before', lrm=1.0):

	if method_type == 'decoder':
		space='euclidean'

	if cuda:
		device = th.device("cuda:0" if torch.cuda.is_available() else "cpu")
	else:
		device = th.device("cpu")

	print(f"Computing on {device}")

	encoder = Encoder(n_inputs=coordinates_from.shape[1], n_outputs=coordinates_to.shape[1], bn=bn)
	encoder = encoder.to(device)
	
	optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=wd)    

	loss = torch.nn.MSELoss()
	
	if tb:
		writer = SummaryWriter()

	X_train, X_test, y_train, y_test = train_test_split(coordinates_from, coordinates_to, test_size=0.3, random_state=42)
	if cuda:
		t_X_train = torch.Tensor(X_train).cuda()
		t_y_train = torch.Tensor(y_train).cuda()
		t_X_test = torch.Tensor(X_test).cuda()
		t_y_test = torch.Tensor(y_test).cuda()
		loader = DataLoader(TensorDataset(torch.Tensor(X_train).cuda(), 
							torch.Tensor(y_train).cuda()),
							batch_size=batch_size,
							# pin_memory=True,
							shuffle=True)
	else:
		t_X_train = torch.Tensor(X_train)
		t_y_train = torch.Tensor(y_train)
		t_X_test = torch.Tensor(X_test)
		t_y_test = torch.Tensor(y_test)
		loader = DataLoader(TensorDataset(torch.Tensor(X_train), 
							torch.Tensor(y_train)),
							batch_size=batch_size,
							# pin_memory=True,
							shuffle=True)
	
	poincare_distances = PoincareDistance()	

	train_error = []
	test_error = []

	pbar = tqdm(range(n_epochs), ncols=80)
	t_start = timeit.default_timer()

	n_iter = 0
	for epoch in pbar:
		epoch_error = 0

		# if epoch == 100:
		# 	optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=wd)

		if epoch == n_warmup:
			optimizerSGD = RiemannianSGD(encoder.parameters(), lr=1e-4)

		for inputs, targets in loader:
			preds = encoder(inputs)

			if epoch >= n_warmup and space =='poincare':
				z = PoincareDistance()(preds, targets)
				error_encoder =  th.mean(z)
				optimizerSGD.zero_grad()    
				error_encoder.backward()
				optimizerSGD.step()
			else:
				error_encoder = loss(preds, targets) 
				optimizer.zero_grad()
				error_encoder.backward()
				optimizer.step()

			epoch_error += error_encoder.item()

			if tb:
				writer.add_scalar("data/train/error", error_encoder.item(), n_iter)
				writer.add_histogram("data/train/predictions", preds.data, n_iter)
				writer.add_histogram("data/train/targets", targets.data, n_iter)

			n_iter += 1
		
		pbar.set_description("loss: {:.5e}".format(epoch_error))
		
		encoder.eval()
		if space =='poincare':
			test_error.append(th.mean(poincare_distances(encoder(t_X_test), t_y_test)).detach().cpu() )
			train_error.append(th.mean(poincare_distances(encoder(t_X_train), t_y_train)).detach().cpu() )
		else:
			test_error.append(loss(encoder(t_X_test), t_y_test).detach().cpu() )
			train_error.append(loss(encoder(t_X_train), t_y_train).detach().cpu()  )		

		if epoch % 100 == 0:
			fig = plt.figure(figsize=(5, 5))
			plt.plot(np.log10(train_error), label='train', color='red')
			plt.plot(np.log10(test_error), label='test', color='green')            
			plt.legend(['train', 'test'])
			plt.show()
			plt.savefig(file_name + '_training_error.png', format='png', dpi=150)

		if tb:
			writer.add_scalar("data/test/epoch_error", test_error[-1], epoch)
			writer.add_scalar("data/train/epoch_error", train_error[-1], epoch)
			writer.add_histogram("data/test/predictions", encoder(t_X_test).detach().cpu(), epoch)
			writer.add_histogram("data/test/targets", t_y_test, epoch)

		encoder.train()

	encoder.eval()

	print(f"epoch_error = {epoch_error:.5e}")
	elapsed = timeit.default_timer() - t_start
	print(f"Time: {elapsed:.2f}")

	print(f"Max norm = {torch.max(torch.sum(encoder(t_X_test)**2, dim=1)):.2f}")
	
	encoder = encoder.to("cpu")
	th.save(encoder.state_dict(), f"{file_name}_{method_type}.pth.tar")
	
	if tb:
		writer.close()

	return encoder



if __name__ == "__main__":  
	parser = argparse.ArgumentParser(description='Poincare maps autoencoder')
	parser.add_argument('--model', help='Model', type=str, default='MyeloidProgenitors')
	parser.add_argument('--method', help='Method', type=str, default='poincare')
	parser.add_argument('--nn', help='Encoder or decoder', type=str, default='decoder')

	parser.add_argument('--epochs', help='Number of epochs', type=int, default=1000)
	parser.add_argument('--cuda', help='Use GPU', type=int, default=0)
	parser.add_argument('--batch_size', help='Batch size', type=int, default=-1)
	parser.add_argument('--lr', help='Learning rate', type=float, default=1e-4)
	parser.add_argument('--lrm', help='Learning rate multiplier', type=float, default=1.0)
	parser.add_argument('--wd', help='Weight decay', type=float, default=1e-4)
	parser.add_argument('--tb', help='Tensor board', type=float, default=0)
	parser.add_argument('--bn', help='Batchnorm', type=str, default='no')

	opt = parser.parse_args()

	if opt.model == 'MyeloidProgenitors':    	
		model_name = 'MyeloidProgenitorsInterv'
		model_name_full = 'Planaria'
	elif opt.model == 'Planaria':    	
		model_name = 'Planaria_wo_pp'
		model_name_full = 'Planaria'
	elif opt.model == 'Olsson':    	
		model_name = 'Olsson_wo_HSPC2'
		model_name_full = 'Olsson'
	elif opt.model == 'Paul7':    	
		model_name = 'Paul_wo_7MEP'
		model_name_full = 'Paul'
	elif opt.model == 'Paul10':    	
		model_name = 'Paul_wo_10GMP'
		model_name_full = 'Paul'
	elif opt.model == 'Paul':    	
		model_name = 'Paul_wo_proj'
		model_name_full = 'Paul'
	elif opt.model == 'PaulEry':    	
		model_name = 'Paul_wo_ery'
		model_name_full = 'Paul'

	with open("predictions/config.txt", "w") as f:
		f.write(opt.__str__())
	
	fin = f"datasets/{model_name}"
	fout = f"decoder/interpolate_{model_name}_lr{opt.lr:.1e}_lrm{opt.lrm:.1e}_wd{opt.wd:.1e}_epochs{opt.epochs:d}_BN{opt.bn}"

	data, true_labels, col_names = read_data(f"datasets/{model_name}", normalize=False, n_pca=0)
	data_ori = np.copy(data)

	if opt.batch_size < 0:
	  opt.batch_size = min(2000, int(np.size(data, 0)/4))
	  print(f"batchsize = {opt.batch_size}")


	st = np.std(data, axis=0)
	st[st == 0] = 1
	m = np.mean(data, axis=0)	
	# m = 0
	# s= 1
	data = (data - m) / st
	
	print(f"{opt.method} {opt.nn}, wd={opt.wd:.1e}")

	poincare_coord = pd.read_csv(f"predictions/{model_name}_PM.csv", sep=',').values

	if opt.method == 'poincare':
		print('poincare')

		plot_embedding(poincare_coord, labels=true_labels, coldict=color_dict,file_name=f"{fout}_{opt.method}_ori")

		if opt.nn == 'decoder':
			decoder = train_model(poincare_coord, data, 
				f"{fout}_poincare", opt.nn,
				n_epochs=opt.epochs, space='euclidean', 
				lr=opt.lr, batch_size=opt.batch_size, wd=opt.wd,
				cuda=opt.cuda, tb=opt.tb, bn=opt.bn, lrm=opt.lrm)

			x_pedicted = decoder(th.Tensor(poincare_coord)).detach().numpy()*st + m
		else:
			decoder = train_model(data, poincare_coord,
				f"{fout}_poincare", opt.nn,
				n_epochs=opt.epochs, space='poincare', 
				lr=opt.lr, batch_size=opt.batch_size, wd=opt.wd,
				cuda=opt.cuda, tb=opt.tb, bn=opt.bn, lrm=opt.lrm)
	else:
		fa_coord = pd.read_csv(f"predictions/{model_name}_{opt.method}.csv", sep=',').values
		plot_embedding(fa_coord, labels=true_labels, coldict=color_dict,
			file_name=f"{fout}_{opt.method}_ori")
		if opt.nn == 'decoder':
			decoder = train_model(fa_coord, data,
				f"{fout}_{opt.method}", opt.nn,
				n_epochs=opt.epochs, space='euclidean', 
				lr=opt.lr, batch_size=opt.batch_size, wd=opt.wd,
				cuda=opt.cuda, tb=opt.tb, bn=opt.bn, lrm=opt.lrm)
			x_pedicted = decoder(th.Tensor(fa_coord)).detach().numpy()*st + m
		else:
			decoder = train_model(data, fa_coord, 
				f"{fout}_{opt.method}", opt.nn,
				n_epochs=opt.epochs, space='euclidean', 
				lr=opt.lr, batch_size=opt.batch_size, wd=opt.wd,
				cuda=opt.cuda, tb=opt.tb, bn=opt.bn, lrm=opt.lrm)
	
	embedding = umap.UMAP(n_neighbors=30).fit_transform(x_pedicted)
	plot_embedding(embedding, labels=true_labels, coldict=color_dict,
		file_name=f"{fout}_{opt.method}_reconstruction_umap")

	# model = PoincareMaps(poincare_coord)
	# ncol = 9
	# model.plot_markers(data_ori[:, :ncol] - x_pedicted[:, :ncol], col_names[:ncol], 
	# 	file_name=f"{fout}_{opt.method}_reconstruction_error", pm_type='ori')