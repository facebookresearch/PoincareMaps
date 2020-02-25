import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from poincare_maps import *
import timeit

from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph

from data import connect_knn


def get_scalars(qs):
	lcmc = np.copy(qs)
	N = len(qs)
	for j in range(N):
		lcmc[j] = lcmc[j] - j/N    
	K_max = np.argmax(lcmc) + 1

	Qlocal = np.mean(qs[:K_max])
	Qglobal = np.mean(qs[K_max:])

	return Qlocal, Qglobal, K_max

def get_rank_high(data, k_neighbours = 15, knn_sym=True):
	# computes ranking of the original dataset through geodesic distances
	KNN = kneighbors_graph(data, k_neighbours,
						   mode='distance', 
						   include_self=False).toarray()
	if knn_sym:
		KNN = np.maximum(KNN, KNN.T)

	n_components, labels = csgraph.connected_components(KNN)
	print(n_components)
	D_high = graph_shortest_path(KNN)

	if n_components:
		max_dist = np.max(D_high)*10
		for comp in np.unique(labels):
			ix_comp = np.where(labels == comp)[0]
			ix_not_comp = np.where(labels != comp)[0]
			for i in ix_comp:
				for j in ix_not_comp:
					D_high[i, j] = max_dist
					D_high[j, i] = max_dist

	Rank_high = get_ranking(D_high)
	
	return Rank_high


# def get_rank_high(data, k_neighbours = 15, knn_sym=True):
# 	# computes ranking of the original dataset through geodesic distances
# 	KNN = kneighbors_graph(data, k_neighbours,
# 						   mode='distance', 
# 						   include_self=False).toarray()
# 	if knn_sym:
# 		KNN = np.maximum(KNN, KNN.T)

# 	n_components, labels = csgraph.connected_components(KNN)

# 	if (n_components > 1):
# 			print('Connecting', n_components)
# 			distances = pairwise_distances(data, metric='euclidean')
# 			KNN = connect_knn(KNN, distances, n_components, labels)
	
# 	D_high = graph_shortest_path(KNN)
# 	Rank_high = get_ranking(D_high)
	
# 	return Rank_high

def get_ranking(D):
	start = timeit.default_timer()
	n = len(D)

	Rank = np.zeros([n, n])
	for i in range(n):
		# tmp = D[i, :10]
		idx = np.array(list(range(n)))
        
		sidx = np.argsort(D[i, :])
		Rank[i, idx[sidx][1:]] = idx[1:]-np.ones(n-1)

	print(f"Ranking: time = {(timeit.default_timer() - start):.1f} sec")
	return Rank

# def get_ranking(D):
# 	start = timeit.default_timer()
# 	n = len(D)

# 	Rank = np.zeros([n, n])
# 	for i in range(n):
# 		# tmp = D[i, :10]
# 		idx = list(range(n))
# 		sidx = np.argsort(D[i, :])
# 		for c, j in enumerate(sidx):
# 			Rank[i, idx[j]] = c
# 	print(f"Ranking: time = {(timeit.default_timer() - start):.3f} sec")
# 	return Rank	


def Tk(M, T = "lower"):    
	c = 0
	for i in range(len(M)):
		for j in range(i+1, len(M)):
			if T == "lower":
				c += M[j, i]
			if T == "upper":
				c += M[i, j]

	return(c)


# def get_coRanking(Rank_high, Rank_low):

# 	n = len(Rank_high)
# 	coRank = np.zeros([n-1, n-1])

# 	for k in range(n-1):
# 		for l in range(n-1):
# 			coRank[k, l] = len(np.where((Rank_high == (k+1)) & (Rank_low == (l+1)))[0])
	
# 	return coRank

def get_coRanking(Rank_high, Rank_low):
	start = timeit.default_timer()
	n = len(Rank_high)
	coRank = np.zeros([n-1, n-1])

	for i in range(n):
		for j in range(n):
			k = int(Rank_high[i, j])
			l = int(Rank_low[i, j])
			if (k > 0) and (l > 0):
				coRank[k-1][l-1] += 1
	
	print(f"Co-ranking: time = {(timeit.default_timer() - start):.2f} sec")
	return coRank


def get_score(Rank_high, Rank_low, fname=None):	
	coRank = get_coRanking(Rank_high, Rank_low)
	start = timeit.default_timer()
	n = len(Rank_high) + 1

	df_score = pd.DataFrame(columns=['Qnx', 'Bnx'])

	Qnx = 0
	Bnx = 0
	for K in range(1, n-1):
		Fk = list(range(K))

		Qnx += sum(coRank[:K, K-1]) + sum(coRank[K-1, :K]) - coRank[K-1, K-1]
		Bnx += sum(coRank[:K, K-1]) - sum(coRank[K-1, :K])

		df_score.loc[len(df_score)] = [Qnx /(K*n), Bnx/(K*n)]

	if not (fname is None):
		df_score.to_csv(fname, sep = ',', index=False)

	# print(df_score.mean()[['Qnx', 'Bnx']])
	Qlocal, Q_global, Kmax = get_scalars(df_score['Qnx'].values)
	print(f"Qlocal = {Qlocal:.2f}, Q_global = {Q_global:.2f}, Kmax = {Kmax}")
	print(f"Time = {(timeit.default_timer() - start):.2f} sec")
	return df_score

def get_quality_metrics(coord_high, coord_low, distance='E', fname=None):
	D_high = pairwise_distances(coord_high)
	
	if distance == 'E':		
		D_low = pairwise_distances(coord_low)

	if distance == 'P':
		print('Poincaré space')		
		model = PoincareMaps(coord_low)
		model.get_distances()
		D_low = model.distances

	Rank_high = get_ranking(D_high)
	print('Rank high')

	Rank_low = get_ranking(D_low)
	print('Rank low')

	df_score = get_score(Rank_high, Rank_low, fname=fname)

	return df_score
