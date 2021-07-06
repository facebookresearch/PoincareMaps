import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from poincare_maps import *

from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph
from data import connect_knn

def get_dist_manifold(data, k_neighbours = 20, knn_sym=True):
    """
    Computes ranking of the original dataset through geodesic distances:
    we estimate KNN graph and find shortest distance on it. The geodesic
    distance between disconnected componenents is set to infinity.
    """
    KNN = kneighbors_graph(
        data, k_neighbours, mode='distance', include_self=False).toarray()
    if knn_sym:
        KNN = np.maximum(KNN, KNN.T)
        
    n_components, labels = csgraph.connected_components(KNN)
    
    if (n_components > 1):
        print('Connecting', n_components)
        distances = pairwise_distances(data, metric='euclidean')
        KNN = connect_knn(KNN, distances, n_components, labels)
    
    D_high = graph_shortest_path(KNN)
    return D_high


def get_ranking(distance_matrix):
    """
    Get ranking from distance matrix: from Supplementary eq. (2)-(3) 
    in Klimovskaia et al.
    """
    # According to this definition, reflexive ranks are set
    # to zero and non-reflexive ranks belong to {1,.., N − 1}.
    n = len(distance_matrix)
    Rank = np.zeros([n, n])
    for i in range(n):
        idx = np.array(list(range(n)))        
        sidx = np.argsort(distance_matrix[i, :])
        Rank[i, idx[sidx][1:]] = idx[1:]

    return Rank


def get_coRanking(Rank_high, Rank_low):
    """
    Computes co-ranking matrix Q from Supplementary eq. (4) in Klimovskaia et al.
    """
    N = len(Rank_high)
    coRank = np.zeros([N-1, N-1])

    for i in range(N):
        for j in range(N):
            k = int(Rank_high[i, j])
            l = int(Rank_low[i, j])
            if (k > 0) and (l > 0):
                coRank[k-1][l-1] += 1
    
    return coRank


def get_score(Rank_high, Rank_low, fname=None):     
    """
    Computes Qnx scores from Supplementary eq. (5) in Klimovskaia et al.
    """
    coRank = get_coRanking(Rank_high, Rank_low)
    N = len(coRank)+1

    df_score = pd.DataFrame(columns=['Qnx', 'Bnx'])
    Qnx = 0
    Bnx = 0
    for K in range(1, N):
        Qnx += sum(coRank[:K, K-1]) + sum(coRank[K-1, :K]) - coRank[K-1, K-1]
        Bnx += sum(coRank[:K, K-1]) - sum(coRank[K-1, :K])
        df_score.loc[len(df_score)] = [Qnx /(K*N), Bnx/(K*N)]

    if not (fname is None):
        df_score.to_csv(fname, sep = ',', index=False)
    
    return df_score


def get_scalars(Qnx):
    """
    Computes scalar scores from Supplementary eq. (6)-(8) in Klimovskaia et al.
    """
    N = len(Qnx) # total length of Qnx is smaller than number of samples
    K_max = 0
    val_max = Qnx[0] - 1/N
    for k in range(1, N):
        if val_max < (Qnx[k] - (k+1)/N):
            val_max = Qnx[k] - (k+1)/N
            K_max = k

    Qlocal = np.mean(Qnx[:K_max+1])
    Qglobal = np.mean(Qnx[K_max:])

    return Qlocal, Qglobal, K_max


def get_quality_metrics(    
    coord_high, 
    coord_low,
    distance='euclidean',    
    setting='manifold',
    fname=None,
    k_neighbours=20,
    verbose=False):
    """
    Implementation of 'Scale-independent quality criteria' from Lee et al.    
    Parameters
    ----------
    coord_high : np.array
        Feature matrix of the sample in the high dimensional space.
    coord_low : np.array
        Low dimensional embedding of the sample.
    distance : str (default: 'euclidean')
        Distance metric to compute distanced between points in low dimendional 
        space. Possible parameters: 'euclidean' or 'poincare'.
    setting: str (default: 'manifold')
        Setting to compute distances in the high dimensional space: 'global'
        distances or distances on the 'manifold' using a k=20 KNN graph.
    fname: str, optional (default: None)
        Name of the file where to save all the information about the metrics.
    verbose: bool (default: False)
        A flag if to print the results of the computations.    
    k_neighbours: int (default: 20)
        k-nearest neighbours for setting
    Returns
    -------
    Qlocal: float
        Quality criteria for local qualities of the embedding.
        Range from 0 (bad) to 1 (good).
    Qglobal: float
        Quality criteria for global qualities of the embedding.
        Range from 0 (bad) to 1 (good).
    Kmax: int
        Kmax defines the split of the QNX curv.
    """

    if setting == 'global':
        D_high = pairwise_distances(coord_high)        
    elif setting == 'manifold':
        D_high = get_dist_manifold(
            coord_high, k_neighbours=k_neighbours, knn_sym=True)
    else:
        raise NotImplementedError

    Rank_high = get_ranking(D_high)

    if distance == 'euclidean':     
        D_low = pairwise_distances(coord_low)
    elif distance == 'poincare':
        model = PoincareMaps(coord_low)
        model.get_distances()
        D_low = model.distances    
    else:
        raise NotImplementedError

    Rank_low = get_ranking(D_low)
    df_score = get_score(Rank_high, Rank_low, fname=fname)

    Qlocal, Qglobal, Kmax = get_scalars(df_score['Qnx'].values)
    if verbose:
        print(f"Qlocal = {Qlocal:.2f}, Qglobal = {Qglobal:.2f}, Kmax = {Kmax}")

    return Qlocal, Qglobal, Kmax
