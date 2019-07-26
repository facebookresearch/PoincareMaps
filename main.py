# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.metrics.pairwise import pairwise_distances
from torch.utils.data import TensorDataset
from sklearn.decomposition import PCA
import numpy as np
import argparse
import torch

from data import create_output_name, prepare_data, compute_rfa
from model import PoincareEmbedding, PoincareDistance
from model import poincare_root, poincare_translation
from rsgd import RiemannianSGD
from train import train
from visualize import *

import timeit

if __name__ == "__main__":
		# parse arguments
		parser = argparse.ArgumentParser(description='Poincare maps')
		parser.add_argument('--dim', help='Embedding dimension', type=int, default=2)

		parser.add_argument('--path', help='Dataset to embed', type=str, default='datasets/')
		parser.add_argument('--dset', help='Dataset to embed', type=str, default='ToggleSwitch')
		parser.add_argument('--dest', help='Write results', type=str, default='results/')

		parser.add_argument('--labels', help='has labels', type=int, default=1)
		parser.add_argument('--normalize', help='Apply z-transform to the data', type=int, default=0)
		parser.add_argument('--pca', help='Apply pca for data preprocessing (if pca=0, no pca)', type=int, default=0)

		parser.add_argument('--distfn', help='Distance function (Euclidean, MFImixSym, MFI, MFIsym)', type=str, default='MFIsym')
		parser.add_argument('--distr', help='Target distribution (laplace, gaussian, student)', type=str, default='laplace')
		parser.add_argument('--lossfn', help='Loss funstion (kl, klSym)', type=str, default='klSym')

		parser.add_argument('--root', help='Get root node from labels', type=str, default="root")
		parser.add_argument('--iroot', help='Index of the root cell', type=int, default=-1)
		parser.add_argument('--rotate', help='Rotate', type=int, default=-1)

		parser.add_argument('--knn', help='Number of nearest neighbours in KNN', type=int, default=15)
		parser.add_argument('--connected', help='Force the knn graph to be connected', type=int, default=1)

		parser.add_argument('--sigma', help='Bandwidth in high dimensional space', type=float, default=1.0)
		parser.add_argument('--gamma', help='Bandwidth in low dimensional space', type=float, default=2.0)

		# optimization parameters
		parser.add_argument('--lr', help='Learning rate', type=float, default=0.01)
		parser.add_argument('--lrm', help='Learning rate multiplier', type=float, default=1.0)
		parser.add_argument('--epochs', help='Number of epochs', type=int, default=10000)
		parser.add_argument('--batchsize', help='Batchsize', type=int, default=-1)
		parser.add_argument('--burnin', help='Duration of burnin', type=int, default=100)

		parser.add_argument('--debugplot', help='Plot intermidiate embeddings every N iterations', type=int, default=0)
		
		parser.add_argument('--cuda', help='Use GPU', type=int, default=0)

		parser.add_argument('--tb', help='Tensor board', type=float, default=0)

		opt = parser.parse_args()
				
		color_dict = None
		if opt.dset == "ToggleSwitch":
				opt.root = "root"
		if "MyeloidProgenitors" in opt.dset:
				opt.root = "root"
		if opt.dset == "krumsiek11_blobs":
				opt.root = "root"
		if "Olsson" in opt.dset:
				opt.root = "HSPC-1"
				color_dict = {'Eryth': '#1F77B4',
				'Gran': '#FF7F0E',
				'HSPC-1': '#2CA02C',
				'HSPC-2': '#D62728',
				'MDP': '#9467BD',
				'Meg': '#8C564B',
				'Mono': '#E377C2',
				'Multi-Lin': '#BCBD22',
				'Myelocyte': '#17BECF'}
		if "Paul" in opt.dset:
				opt.root = "root"
				if opt.dset == 'Paul_wo_proj':
					opt.root = "6Ery"
				color_dict = {'12Baso': '#0570b0', '13Baso': '#034e7b',
					 '11DC': '#ffff33', 
					 '18Eos': '#2CA02C', 
					 '1Ery': '#fed976', '2Ery': '#feb24c', '3Ery': '#fd8d3c', '4Ery': '#fc4e2a', '5Ery': '#e31a1c', '6Ery': '#b10026',
					 '9GMP': '#999999', '10GMP': '#4d4d4d',
					 '19Lymph': '#35978f', 
					 '7MEP': '#E377C2', 
					 '8Mk': '#BCBD22', 
					 '14Mo': '#4eb3d3', '15Mo': '#7bccc4',
					 '16Neu': '#6a51a3','17Neu': '#3f007d',
					 'root': '#000000'}
		if opt.dset == "Moignard2015":
				opt.root = "PS"
		if "Planaria" in opt.dset:
			opt.root = "neoblast 1"
			color_dict = {'neoblast 1': '#CCCCCC',
			'neoblast 2': '#7f7f7f',
			'neoblast 3': '#E6E6E6',
			'neoblast 4': '#D6D6D6',
			'neoblast 5': '#C7C7C7',
			'neoblast 6': '#B8B8B8',
			'neoblast 7': '#A8A8A8',
			'neoblast 8': '#999999',
			'neoblast 9': '#8A8A8A',
			'neoblast 10':  '#7A7A7A',
			'neoblast 11':  '#6B6B6B',
			'neoblast 12':  '#5C5C5C',
			'neoblast 13':  '#4D4D4D',
			'epidermis DVb neoblast': 'lightsteelblue',
			'pharynx cell type progenitors':  'slategray',
			'spp-11+ neurons':  '#CC4C02',
			'npp-18+ neurons':  '#EC7014',
			'otf+ cells 1': '#993404',
			'ChAT neurons 1': '#FEC44F',
			'neural progenitors': '#FFF7BC',
			'otf+ cells 2': '#662506',
			'cav-1+ neurons': '#eec900',
			'GABA neurons': '#FEE391',
			'ChAT neurons 2': '#FE9929',
			'muscle body':  'firebrick',
			'muscle pharynx': '#CD5C5C',
			'muscle progenitors': '#FF6347',
			'secretory 1':  'mediumpurple',
			'secretory 3':  'purple',
			'secretory 4':  '#CBC9E2',
			'secretory 2':  '#551a8b',
			'early epidermal progenitors':  '#9ECAE1',
			'epidermal neoblasts':  '#C6DBEF',
			'activated early epidermal progenitors':  'lightblue',
			'late epidermal progenitors 2': '#4292C6',
			'late epidermal progenitors 1': '#6BAED6',
			'epidermis DVb':  'dodgerblue',
			'epidermis':  '#2171B5',
			'pharynx cell type': 'royalblue',
			'protonephridia': 'pink',
			'ldlrr-1+ parenchymal cells': '#d02090',
			'phagocytes': 'forestgreen',
			'aqp+ parenchymal cells': '#cd96cd',
			'pigment': '#cd6889',
			'pgrn+ parenchymal cells':  'mediumorchid',
			'psap+ parenchymal cells':  'deeppink',
			'glia': '#cd69c9',
			'goblet cells': 'yellow',
			'parenchymal progenitors':  'hotpink',
			'psd+ cells': 'darkolivegreen',
			'gut progenitors':  'limegreen',
			'interpolation': '#636363',
			'interpolation2': '#999999'}
		

		if opt.dset == 'WagnerScience2018':
			cell_cell_edges = np.genfromtxt('datasets/cell_cell_edges.csv', dtype=int, delimiter=',')
			cell_cell_edges -= 1  # our indexing starts with 0 as common in Python, C etc.
			from scipy.sparse import coo_matrix
			rows = cell_cell_edges[:, 0]
			cols = cell_cell_edges[:, 1]
			length = len(obs)
			ones = np.ones(len(rows), np.uint32)
			connectivities = coo_matrix((ones, (rows, cols)), shape=(length, length))
			# make sure it's symmsetric
			connectivities = connectivities + connectivities.T
			connectivities = connectivities.toarray

			from scipy.sparse import csgraph

			L = csgraph.laplacian(connectivities, normed=False)
			RFA = torch.Tensor(np.linalg.inv(L + np.eye(L.shape[0])))

		else:

			# read and preprocess the dataset
			features, labels = prepare_data(opt.path + opt.dset,
																			with_labels=opt.labels,
																			normalize=opt.normalize,
																			n_pca=opt.pca)

			# compute matrix of RFA similarities
			RFA = compute_rfa(features,
												k_neighbours=opt.knn,
												distfn=opt.distfn,
												connected=opt.connected,
												sigma=opt.sigma)

		if opt.batchsize < 0:
			opt.batchsize = min(2000, int(len(RFA)/8))
			# if opt.dset == "Moignard2015":
			#     opt.batchsize = 1500
		opt.lr = opt.batchsize / 16 * opt.lr


		itlename, fout = create_output_name(opt)

			# PCA of RFA baseline
			# pca_baseline = PCA(n_components=2).fit_transform(RFA)
			# plot2D(pca_baseline.T,
			#        labels,
			#        fout + '_PCARFA',
			#        'PCA of RFA\n' + titlename)

		 
			# build the indexed RFA dataset 
		indices = torch.arange(len(RFA))
		if opt.cuda:
				indices = indices.cuda()
				RFA = RFA.cuda()

		dataset = TensorDataset(indices, RFA)

		# instantiate our Embedding predictor
		predictor = PoincareEmbedding(len(dataset),
																	opt.dim,
																	dist=PoincareDistance,
																	max_norm=1,
																	Qdist=opt.distr, 
																	lossfn = opt.lossfn,
																	gamma=opt.gamma,
																	cuda=opt.cuda)

		# instantiate the Riemannian optimizer 
		start = timeit.default_timer()
		optimizer = RiemannianSGD(predictor.parameters(), lr=opt.lr)

		# train predictor
		embeddings, loss = train(predictor,
														 dataset,
														 optimizer,
														 opt,
														 fout=fout,
														 labels=labels,
														 tb=opt.tb)

		np.savetxt(fout + '.csv', embeddings, delimiter=",")

		titlename = f"loss = {loss:.3e}\ntime = {(timeit.default_timer() - start)/60:.3f} min"
		color_dict = plotPoincareDisc(embeddings.T,
												 labels,
												 fout,
												 titlename,
												 color_dict=color_dict)

		# rotation
		root_hat = poincare_root(opt, labels, features)   
		if root_hat != -1:
				titlename = '{0}\nloss = {1:.3e} rotated'.format(titlename, loss)

				poincare_coord_new = poincare_translation(
						-embeddings[root_hat, :], embeddings)

				plot_poincare_disc(poincare_coord_new,
													 labels=labels,
													 coldict=color_dict,
													 file_name=fout + '_rotated')
