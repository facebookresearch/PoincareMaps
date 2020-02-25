# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from sklearn.cluster import *
# import scanpy.api as sc

from model import poincare_translation
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.decomposition import PCA

from visualize import *
from model import *
import seaborn as sns; sns.set()
import torch as th
from fastdtw import fastdtw
from coldict import *
from scipy.spatial.distance import euclidean
import scanpy.api as sc


sns.set_style('white', {'legend.frameon':True})

colors_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                  '#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B',
                  '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                  '#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B',
                  '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
                  '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
                  '#CCEBC5', '#FFED6F', '#edf8b1', '#c7e9b4', '#7fcdbb',
                  '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']


# colors_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#BCBD22', 
#         '#17BECF', '#40004b', '#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', '#f7f7f7', '#d9f0d3', 
#         '#a6dba0', '#5aae61', '#1b7837', '#00441b', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', 
#         '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f', '#CCEBC5', '#FFED6F', '#edf8b1', '#c7e9b4', '#7fcdbb',
#                   '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']

def plot_poincare_disc(x, labels=None, title_name=None,
    labels_name='labels', labels_order=None, labels_pos=None, labels_text=None,
                       file_name=None, coldict=None,
                       d1=4.5, d2=4.0, fs=9, ms=20,
                       u=None, v=None, alpha=1.0,
                       col_palette=plt.get_cmap("tab10"), print_labels=True,
                       bbox=(1.3, 0.7), leg=True, ft='pdf'):    

    idx = np.random.permutation(len(x))
    df = pd.DataFrame(x[idx, :], columns=['pm1', 'pm2'])
    
    fig = plt.figure(figsize=(d1, d2))
    ax = plt.gca()
    circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)
    if title_name:
        ax.set_title(title_name, fontsize=fs)

    if not (labels is None):
        df[labels_name] = labels[idx]
        if labels_order is None:
            labels_order = np.unique(labels)        
        if coldict is None:
            coldict = dict(zip(labels_order, col_palette[:len(labels)]))
        sns.scatterplot(x="pm1", y="pm2", hue=labels_name, 
                        hue_order=labels_order,
                        palette=coldict,
                        alpha=alpha, edgecolor="none",
                        data=df, ax=ax, s=ms)
        
        if leg:
            ax.legend(fontsize=fs, loc='outside', bbox_to_anchor=bbox, facecolor='white')
        else:
            ax.legend_.remove()
            
    else:
        sns.scatterplot(x="pm1", y="pm2",
                        data=df, ax=ax, s=ms)

        if leg == False:
            ax.legend_.remove()

    if not (u is None):     
        a, b = get_geodesic_parameters(u, v)        
        circle_geo = plt.Circle((-a/2, -b/2), radius=np.sqrt(a**2/4 + b**2/4 - 1),  fc='none', color='grey')
        ax.add_patch(circle_geo)

    fig.tight_layout()
    ax.axis('off')
    ax.axis('equal') 

    if print_labels:
        if labels_text is None:
            labels_list = np.unique(labels)
        else:
            labels_list = np.unique(labels_text)
        if labels_pos is None:  
            labels_pos = {}  
            for l in labels_list:
        #         i = np.random.choice(np.where(labels == l)[0])
                ix_l = np.where(labels == l)[0]
                Dl = poincare_distance(th.DoubleTensor(x[ix_l, :])).numpy()
                i = ix_l[np.argmin(Dl.sum(axis=0))]
                labels_pos[l] = i

        for l in labels_list:    
            ax.text(x[labels_pos[l], 0], x[labels_pos[l], 1], l, fontsize=fs)

    ax.set_ylim([-1.01, 1.01])
    ax.set_xlim([-1.01, 1.01]) 

    plt.tight_layout()

    if file_name:
        if ft == 'png':            
            plt.savefig(file_name + '.' + ft, format=ft, dpi=300)
        else:
            plt.savefig(file_name + '.' + ft, format=ft)

    return labels_pos

class PoincareMaps:
    def __init__(self, coordinates, cpalette=None):
        self.coordinates = coordinates
        self.distances = None       
        self.radius = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
        self.iroot = np.argmin(self.radius)
        self.labels_pos = None
        if cpalette is None:
            self.colors_palette = colors_palette
        else:
            self.colors_palette = cpalette
        
    def find_iroot(self, labels, head_name):
        head_idx = np.where(labels == head_name)[0]
        if len(head_idx) > 1:
            D = self.distances[head_idx, :][head_idx]
            self.iroot = head_idx[np.argmin(D.sum(axis=0))]
        else:
            self.iroot = head_idx[0]            

    def get_distances(self):
        self.distances = poincare_distance(th.DoubleTensor(self.coordinates)).numpy()

    def rotate(self):
        self.coordinates_rotated = poincare_translation(-self.coordinates[self.iroot, :], self.coordinates)     

    def plot(self, pm_type='ori', labels=None, 
        labels_name='labels', print_labels=True, labels_text=None,
        labels_order=None, coldict=None, file_name=None, title_name=None, alpha=1.0,
        zoom=None, show=True, d1=4.5, d2=4.0, fs=9, ms=20, bbox=(1.3, 0.7), u=None, v=None, leg=True, ft='pdf'):                            
        if pm_type == 'ori':
            coordinates = self.coordinates
        
        elif pm_type == 'rot':
            coordinates = self.coordinates_rotated

        if labels_order is None:
            labels_order = np.unique(labels)

        if not (zoom is None):
            if zoom == 1:
                coordinates = np.array(linear_scale(coordinates))
            else:           
                radius = np.sqrt(coordinates[:,0]**2 + coordinates[:,1]**2)
                idx_zoom = np.where(radius <= 1/zoom)[0]
                coordinates = coordinates[idx_zoom, :]
                coordinates = np.array(linear_scale(coordinates))    
                coordinates[np.isnan(coordinates)] = 0
                labels = labels[idx_zoom]

        
        self.labels_pos = plot_poincare_disc(coordinates, title_name=title_name, 
            print_labels=print_labels, labels_text=labels_text,
            labels=labels, labels_name=labels_name, labels_order=labels_order, labels_pos = self.labels_pos,
                       file_name=file_name, coldict=coldict, u=u, v=v, alpha=alpha,
                       d1=d1, d2=d2, fs=fs, ms=ms, col_palette=self.colors_palette, bbox=bbox, leg=leg, ft=ft)

    def detect_lineages(self, n_lin=2, clustering_name='spectral', k=15, rotated=False):
        pc_proj = []

        if rotated:
            x = self.coordinates_rotated
        else:
            x = self.coordinates
        
        for i in range(len(x)):
            pc_proj.append(get_projected_coordinates(x[i]))
        
        if clustering_name == 'spectral':
            clustering = SpectralClustering(n_clusters=n_lin, eigen_solver='arpack', affinity="nearest_neighbors", n_neighbors=k).fit(pc_proj)      
        elif clustering_name == 'dbs':
            clustering = DBSCAN(eps=1/180, min_samples=10).fit(pc_proj)
        elif clustering_name == 'kmeans':
            clustering = KMeans(n_clusters=n_lin).fit(pc_proj)
        else:
            clustering = AgglomerativeClustering(linkage='ward', n_clusters=n_lin).fit(pc_proj)

        self.lineages = clustering.labels_

    def detect_cluster(self, n_clusters=2, clustering_name='spectral', k=15):
        if clustering_name == 'spectral':
            similarity = np.exp(-self.distances**2)
            clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed', n_neighbors=k).fit(similarity)
        else:
            clustering = AgglomerativeClustering(linkage='average', n_clusters=n_clusters, affinity='precomputed').fit(self.distances**2)

        self.clusters = clustering.labels_


    def plot_distances(self, cell=None, pm_type='rot', ss=10, eps=4.0, file_name=None, title_name=None, idx_zoom=None, show=False, fs=8, ms=3):
        if cell is None:
            cell = self.iroot
            
        if pm_type == 'ori':
            coordinates = self.coordinates
        elif pm_type == 'rot':
            coordinates = self.coordinates_rotated

        fig = plt.figure(figsize=(5, 5))
        circle = plt.Circle((0, 0), radius=1, color='black', fc="None")    
        cm = plt.cm.get_cmap('rainbow')
        
        mycmap = np.minimum(list(self.distances[:, cell]), eps)
        
        plt.gca().add_patch(circle)
        plt.plot(0, 0, 'x', c=(0, 0, 0), ms=ms)
        if title_name:
            plt.title(title_name, fontsize=fs)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c=mycmap, s=ss, cmap=cm)
        plt.plot(coordinates[cell, 0], coordinates[cell, 1], 'd', c='red')

        plt.plot(0, 0, 'x', c=(1, 1, 1), ms=ms)    
        plt.axis('off')
        plt.axis('equal')        
        
        plt.axis('off')
        plt.axis('equal')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=fs) 


        if file_name:
            plt.savefig(file_name + '.pdf', format='pdf')
        
        plt.show()
        plt.close(fig)


    def plot_distances_between_clusters(self, labels, pm_type='rot', eps = 4.0, file_name=None, fs=9):
        if pm_type == 'ori':
            poincare_coord = self.coordinates
        elif pm_type == 'rot':
            poincare_coord = self.coordinates_rotated

        cell_list = np.unique(labels)
        n = len(labels)    
        
        n_plt = len(cell_list)
        # n2 = int(np.sqrt(n_plt))
        n2 = 3
        n1 = n_plt // n2
        
        if n1*n2 < n_plt:
            n1 += 1

        if n1 == 1:
            n1 = 2
        
        if n2 == 1:
            n2 = 2

        f, axs = plt.subplots(n1, n2, sharey=False, figsize=(n2*3, n1*3))

        l=0
        for i in range(n1):
            for j in range(n2):
                if l < n_plt:
                    cell = np.random.choice(np.where(labels == cell_list[l])[0])

                    mycmap = self.distances[cell]

                    circle = plt.Circle((0, 0), radius=1, color='black', fc="None")
                    axs[i, j].add_patch(circle)
                    axs[i, j].axis('off')
                    axs[i, j].axis('equal')
                    axs[i, j].plot(0, 0, 'x', c=(0, 0, 0), ms=6)
                    cm = plt.cm.get_cmap('rainbow')
                    sc = axs[i, j].scatter(poincare_coord[:, 0], poincare_coord[:, 1], c=mycmap, s=15, cmap=cm)    
                    axs[i, j].set_title(cell_list[l], fontsize=fs)
                    axs[i, j].plot(poincare_coord[cell, 0], poincare_coord[cell, 1], 'd', c='red')
                else:
                    axs[i, j].axis('off')
                    axs[i, j].axis('equal')
                l+=1
        if file_name:
            plt.savefig(file_name + '.pdf', format='pdf')
        plt.show()
        plt.close(f)

    def plot_markers(self, data, markesnames, pm_type='rot', file_name=None, fs=8, sc=3):
        if pm_type == 'ori':
            poincare_coord = self.coordinates
        elif pm_type == 'rot':
            poincare_coord = self.coordinates_rotated

        n_plt = np.size(data, 1)    
        
        # n2 = int(np.sqrt(n_plt))
        n2 = 3
        n1 = n_plt // n2
        
        if n1*n2 < n_plt:
            n1 += 1

        if n1 == 1:
            n1 = 2
        
        if n2 == 1:
            n2 = 2 
        
        f, axs = plt.subplots(n1, n2, sharey=False, figsize=(n2*sc, n1*sc))

        cm = plt.cm.get_cmap('jet')

        l=0
        for i in range(n1):
            for j in range(n2):            
                axs[i, j].axis('off')
                axs[i, j].axis('equal')            
                if l < n_plt:
                    circle = plt.Circle((0, 0), radius=1, color='black', fc="none")
                    axs[i, j].add_patch(circle)
                    axs[i, j].plot(0, 0, 'x', c=(0, 0, 0), ms=3)
                    axs[i, j].axis('equal')                 
                    sc = axs[i, j].scatter(poincare_coord[:, 0], poincare_coord[:, 1], c=data[:, l], s=5, cmap=cm)    
                    axs[i, j].set_title(markesnames[l], fontsize=fs)
                    plt.colorbar(sc,ax=axs[i,j])

                    if l == n_plt:
                        axs[i, j].legend(np.unique(labels), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs)
                l+=1

        if file_name:
            plt.savefig(file_name + '.pdf', format='pdf')


        plt.show()
        plt.close(f)

    def plot_markers_radius(self, data, markesnames, labels, file_name=None, fs=8):
        pm_pseudotime = np.sqrt(self.coordinates_rotated[:,0]**2 + self.coordinates_rotated[:,1]**2)

        n_plt = len(markesnames)
        # n2 = int(np.sqrt(n_plt))
        n2 = 2
        n1 = n_plt // n2
        if n1*n2 < n_plt:
            n1 += 1

        if n1 == 1:
            n1 = 2
        
        if n2 == 1:
            n2 = 2

        fig, axs = plt.subplots(n1, n2, sharex=True, sharey=False, figsize=(n2*4 + 3, n1*4))

        i = 0
        for i1 in range(n1):
            for i2 in range(n2):
                axs[i1, i2].grid('off')
                axs[i1, i2].axis('equal')
                axs[i1, i2].yaxis.set_tick_params(labelsize=fs)
                axs[i1, i2].xaxis.set_tick_params(labelsize=fs)
                if i < n_plt:
                    marker = markesnames[i]
                    for j, label in enumerate(np.unique(labels)):
                        idx = np.where(labels == label)[0]
                        axs[i1, i2].plot(pm_pseudotime[idx], data[idx, i], marker='o', markerfacecolor='none', c=self.colors_palette[j], linestyle='', ms=2, label=marker)                
                    axs[i1, i2].set_title(marker, fontsize=fs)
                    i += 1
                    if i == n_plt:
                        axs[i1, i2].legend(np.unique(labels), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs)
                else:
                    axs[i1, i2].axis('off')     

        for ax in axs.flat:
            ax.set(xlabel='radius')
            # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        if file_name:
            plt.savefig(file_name + '.pdf', format='pdf')

        plt.show()
        plt.close(fig)


    def plot_pseudotime(self, data, markesnames, labels, file_name=None, fs=8, idx=[], pm_pseudotime=None,colors_dict=None):        
        if pm_pseudotime is None:
            pm_pseudotime = self.distances[self.iroot, :]

        if colors_dict is None:
            colors_dict = dict(zip(np.unique(labels), self.colors_palette[:len(np.unique(labels))]))
        
        if len(idx):
            # preserve global colors                            
            pm_pseudotime = pm_pseudotime[idx]
            data = data[idx, :]
            labels = labels[idx]

        n_plt = len(markesnames)
        # n2 = int(np.sqrt(n_plt))
        n2 = 3
        n1 = n_plt // n2

        if n1*n2 < n_plt:
            n1 += 1

        if n1 == 1:
            n1 = 2
        
        if n2 == 1:
            n2 = 2


        pl_size = 2
        fig, axs = plt.subplots(n1, n2, sharey=False, figsize=(n2*pl_size + 2, n1*pl_size))

        i = 0               

        for i1 in range(n1):
            for i2 in range(n2):
                axs[i1, i2].grid('off')
                axs[i1, i2].yaxis.set_tick_params(labelsize=fs)
                axs[i1, i2].xaxis.set_tick_params(labelsize=fs)
                if i < n_plt:
                    marker = markesnames[i]
                    for j, label in enumerate(np.unique(labels)):
                        idx = np.where(labels == label)[0]
                        axs[i1, i2].plot(pm_pseudotime[idx], data[idx, i], marker='o', markerfacecolor='none', c=colors_dict[label], linestyle='', ms=2, label=marker)                
                    axs[i1, i2].set_title(marker, fontsize=fs)
                    i += 1
                    if i == n_plt:
                        axs[i1, i2].legend(np.unique(labels), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs)
                else:
                    axs[i1, i2].axis('off')
        axs[i1, i2].legend(np.unique(labels), loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fs)
        
        # for ax in axs.flat:
        #   ax.set(xlabel='pseudotime')
        #   # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #   ax.label_outer()

        plt.xlabel('pseudotime', fontsize=fs)
        fig.tight_layout()

        if file_name:
            plt.savefig(file_name + '.pdf', format='pdf')

        plt.show()
        plt.close(fig)





def get_geodesic_parameters(u, v, eps=1e-10):
    if all(u) == 0:
        u = np.array([eps, eps])
    if all(v) == 0:
        v = np.array([eps, eps])

    nu = u[0]**2 + u[1]**2
    nv = v[0]**2 + v[1]**2
    a = (u[1]*nv - v[1]*nu + u[1] - v[1]) / (u[0]*v[1] - u[1]*v[0])
    b = (v[0]*nu - u[0]*nv + v[0] - u[0]) / (u[0]*v[1] - u[1]*v[0])
    return a, b

def intermediates(p1, p2, nb_points=20):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return np.array([[p1[0] + i * x_spacing, p1[1] +  i * y_spacing] 
            for i in range(1, nb_points+1)])
    
    
def poincare_linspace(u, v, n_points=175):            
    if (np.sum(u**2) == 0):
        x = np.linspace(u[0], v[0], num=n_points)
        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] = x 
        if v[0] != 0:
            k = v[1]/v[0]
            interpolated[:, 1] = k*interpolated[:, 0]
        else:
            interpolated[:, 1] = np.linspace(0, v[1], num=n_points)
    elif (np.sum(v**2) == 0):
        x = np.linspace(u[0], v[0], num=n_points)
        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] = x 
        if u[0] != 0:
            k = u[1]/u[0]
            interpolated[:, 1] = k*interpolated[:, 0]
        else:
            interpolated[:, 1] = np.linspace(0, u[1], num=n_points)
                
    else:
        a, b = get_geodesic_parameters(u, v)

        x = np.linspace(u[0], v[0], num=n_points)

        interpolated = np.zeros([len(x), 2])
        interpolated[:, 0] =x 

        r = a**2/4 + b**2/4 - 1
        y_1 = -b/2 + np.sqrt(r - (x+a/2)**2)
        y_2 = -b/2 - np.sqrt(r - (x+a/2)**2)

        if max(x**2 + y_1**2) > 1:
            interpolated[:, 1] = y_2 
        elif max(x**2 + y_2**2) > 1:
            interpolated[:, 1] = y_1
        elif (np.mean(y_1) <= max(u[1], v[1])) and (np.mean(y_1) >= min(u[1], v[1])):
            interpolated[:, 1] = y_1
        else:
            interpolated[:, 1] = y_2

    return interpolated



def init_scanpy(data, col_names, head_name, true_labels, fin, k=30, n_pcs=20, computeEmbedding=True):
    head_idx = np.where(true_labels == head_name)[0]
    if len(head_idx) > 1:
        D = pairwise_distances(data[head_idx, :], metric='euclidean')
        iroot = head_idx[np.argmin(D.sum(axis=0))]
    else:
        iroot = head_idx[0]
        
    adata = sc.AnnData(data)
    adata.var_names = col_names
    adata.obs['labels'] = true_labels
    adata.uns['iroot'] = iroot
    if computeEmbedding:
        if n_pcs:
            sc.pp.pca(adata, n_comps=n_pcs)
            sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_pcs)
        else:
            sc.pp.neighbors(adata, n_neighbors=k)
        
    
        sc.tl.louvain(adata, resolution=0.9)
        louvain_labels = np.array(list(adata.obs['louvain']))
        
        sc.tl.paga(adata)
        sc.tl.draw_graph(adata)
        sc.tl.diffmap(adata)
        sc.tl.tsne(adata)
        sc.tl.umap(adata)
        sc.tl.pca(adata, n_comps=2)

        sc.pl.paga(adata)
        sc.tl.draw_graph(adata, init_pos='paga')
    else:
        louvain_labels = []

    sc.settings.figdir = fin
    sc.settings.autosave = True
    # sc.settings.set_figure_params(dpi=80, dpi_save=300, color_map='Set1', format='pdf')
    sc.settings.set_figure_params(dpi=80, dpi_save=300, format='pdf')
        
    return adata, iroot, louvain_labels


def plotBenchamrk(adata, true_labels, fname_benchmark, method='X_draw_graph_fa', pl_size=2.4, n1=3, n2=3, ms=10, fs=9, coldict=None):
    labels_order=np.unique(true_labels)
    if coldict is None:
        colors_palette = get_palette(coldict)
        coldict = dict(zip(labels_order, colors_palette[:len(labels_order)]))

    fig = plt.figure(figsize=(n2*pl_size, n1*pl_size))
    ax = plt.gca()

    title_name_dict = {'X_pca': 'PCA',
                        'X_tsne': 'tSNE',
                        'X_umap': 'UMAP', 
                       'X_diffmap': 'DiffusionMaps', 
                       'X_draw_graph_fa': 'ForceAtlas2'}


    title_name=title_name_dict[method]
    axs_names=['x1', 'x2']
    if method == 'X_diffmap':
        x=adata.obsm[method][:, 1:3]
    else:
        x=adata.obsm[method]

    idx = np.random.permutation(len(x))
    df = pd.DataFrame(x[idx, :], columns=axs_names)
    df['labels'] = true_labels[idx]
    plt.title(title_name, fontsize=fs)
    ax.axis('equal')
    ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])
    sns.scatterplot(x=axs_names[0], y=axs_names[1], hue='labels', 
                            hue_order=labels_order,
                            palette=coldict,
                            alpha=1.0, edgecolor="none",
                            data=df, ax=ax, s=ms)
    ax.set_xlabel(axs_names[0], fontsize=fs)
    ax.set_ylabel(axs_names[1], fontsize=fs)
    ax.legend_.remove()
    fig.tight_layout()


    fig.tight_layout()        
    plt.savefig( f"{fname_benchmark}_{title_name}.pdf", format='pdf')
    plt.show()


def plotBenchamrks(adata, true_labels, fname_benchmark, pl_size=2.4, n1=2, n2=3, ms=3, fs=9, coldict=None, methods=['X_pca', 'X_umap', 'X_draw_graph_fa']):
    labels_order=np.unique(true_labels)
    if coldict is None:
        coldict = dict(zip(labels_order, colors_palette[:len(labels_order)]))

    fig, axs = plt.subplots(n1, n2, sharex=False, sharey=False, figsize=(n2*pl_size, n1*pl_size))
    methods=['X_pca', 'X_tsne', 'X_umap', 'X_diffmap', 'X_draw_graph_fa']
    title_name_dict = {'X_pca': 'PCA',
                        'X_tsne': 'tSNE',
                        'X_umap': 'UMAP', 
                       'X_diffmap': 'DiffusionMaps', 
                       'X_draw_graph_fa': 'ForceAtlas2'}

    l=0
    for i in range(n1):
        for j in range(n2):
            if l < len(methods):
                method=methods[l]
                title_name=title_name_dict[method]
                axs_names=['x1', 'x2']
                if method == 'X_diffmap':
                    x=adata.obsm[method][:, 1:3]
                else:
                    x=adata.obsm[method]
                idx = np.random.permutation(len(x))
                df = pd.DataFrame(x[idx, :], columns=axs_names)
                df['labels'] = true_labels[idx]
                axs[i, j].set_title(title_name, fontsize=fs)
                axs[i, j].axis('equal')
                axs[i, j].grid('off')
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                sns.scatterplot(x=axs_names[0], y=axs_names[1], hue='labels', 
                                        hue_order=labels_order,
                                        palette=coldict,
                                        alpha=1.0, edgecolor="none",
                                        data=df, ax=axs[i, j], s=ms)
                axs[i, j].set_xlabel(axs_names[0], fontsize=fs)
                axs[i, j].set_ylabel(axs_names[1], fontsize=fs)
                axs[i, j].legend_.remove()
                fig.tight_layout()
            else:
                axs[i, j].axis('off')
                axs[i, j].grid('off')
                axs[i, j].yaxis.set_tick_params(labelsize=fs)
                axs[i, j].xaxis.set_tick_params(labelsize=fs)
            l += 1
    fig.tight_layout()        
    plt.savefig(fname_benchmark + 'benchmarks.pdf', format='pdf')
    plt.show()
    


def read_data(fin, with_labels=True, normalize=False, n_pca=20):
    """
    Reads a dataset in CSV format from the ones in datasets/
    """
    df = pd.read_csv(fin + '.csv', sep=',')
    n = len(df.columns)

    if with_labels:
        x = np.double(df.values[:, 0:n - 1])
        labels = df.values[:, (n - 1)]
        labels = labels.astype(str)
        colnames = df.columns[0:n - 1]
    else:
        x = np.double(df.values)
        labels = ['unknown'] * np.size(x, 0)
        colnames = df.columns

    n = len(colnames)

    idx = np.where(np.std(x, axis=0) != 0)[0]
    x = x[:, idx]

    if normalize:
        s = np.std(x, axis=0)
        s[s == 0] = 1
        x = (x - np.mean(x, axis=0)) / s

    if n_pca:
        if n_pca == 1:
            n_pca = n

        nc = min(n_pca, n)
        pca = PCA(n_components=nc)
        x = pca.fit_transform(x)

    labels = np.array([str(s) for s in labels])

    return x, labels, colnames

def euclidean_distance(x):
    th.set_default_tensor_type('torch.DoubleTensor')
    # print('computing euclidean distance...')
    nx = x.size(0)
    x = x.contiguous()
    
    x = x.view(nx, -1)

    norm_x = th.sum(x ** 2, 1, keepdim=True).t()
    ones_x = th.ones(nx, 1)

    xTx = th.mm(ones_x, norm_x)
    xTy = th.mm(x, x.t())
    
    d = (xTx.t() + xTx - 2 * xTy)
    d[d < 0] = 0

    return d

def poin_dist(u, v):
    eps = 1e-5
    boundary = 1 - eps
    squnorm = th.clamp(th.sum(u * u, dim=-1), 0, boundary)
    sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, boundary)
    sqdist = th.sum(th.pow(u - v, 2), dim=-1)
    x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
    # arcosh
    z = th.sqrt(th.pow(x, 2) - 1)
    return th.log(x + z)

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


def get_projected_coordinates(u):
#     ax + by = 0
#  y = cx, where c = -a/b = y/x, if b != 0
    if u[0] == 0 or u[1] == 0:
        return [np.sign(u[0]), np.sign(u[1])]
    c = u[1] / u[0]
    x_c = np.sign(u[0])*np.sqrt(1 / (1+c**2))
    y_c = c*x_c
    
    return [x_c, y_c]

def linear_scale(embeddings):
    # embeddings = np.transpose(embeddings)
    sqnorm = np.sum(embeddings ** 2, axis=1, keepdims=True)    
    dist = np.arccosh(1 + 2 * sqnorm / (1 - sqnorm))
    dist = np.sqrt(dist)
    dist /= dist.max()

    sqnorm[sqnorm==0] = 1
    embeddings = dist * embeddings / np.sqrt(sqnorm)
    # embeddings[abs(embeddings) > 1] = 1
    # embeddings = embeddings / np.sum(embeddings ** 2, axis=1, keepdims=True).max()
    return embeddings


def get_confusion_matrix(classes, true_labels_oder, true_labels, fname='', title='Confusion matrix'):
    cm = np.zeros([len(np.unique(classes)), len(np.unique(true_labels))])
    for il, l in enumerate(np.unique(classes)):
        idx = np.where(classes == l)[0]
        for it, tl in enumerate(true_labels_oder):
            cm[il, it] = len(idx[np.where(true_labels[idx] == tl)[0]])
#     classes = [ str(l) for l in np.unique(model.lineages)]
    plot_confusion_matrix(cm, np.unique(classes), true_labels_oder,
                          normalize=True,
                          title=title,
                          cmap=plt.cm.Blues, fname=fname)
    return cm

import itertools
def plot_confusion_matrix(cm, classes, true_labels,
                          normalize=False,
                          title='Lineage matrix',
                          cmap=plt.cm.Blues, fs=9, fname=''):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    
    fig = plt.figure(figsize=(3.5, 3.5))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title, fontsize=fs)
#     plt.colorbar()
    x_tick_marks = np.arange(len(true_labels))
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, true_labels, rotation=45, fontsize=fs)
    plt.yticks(y_tick_marks, classes, fontsize=fs)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fs)

    plt.tight_layout()
    plt.grid(False)

    plt.ylabel(title, fontsize=fs)
    plt.xlabel('stages', fontsize=fs)
    
    if fname != '':
        plt.savefig(fname + '.pdf', format='pdf')
        
    plt.show()
    plt.close()


def get_closest(coordinates, distances, labels, 
                cluster_from='1', 
                cluster_to='5', seed=42):    
    np.random.seed(seed=seed)

    idx_from = random.choice(np.where(labels == cluster_from)[0])
    x1 = coordinates[idx_from]    
    idx_to_all = np.where(labels == cluster_to)[0]
    ix_to = random.choice(idx_to_all)
#     ix_to = idx_to_all[np.argmin(distances[idx_from][idx_to_all])]
    x2 = coordinates[ix_to]
    return x1, x2, idx_from, ix_to

def get_medoids(distances, labels, cluster):
    idx = np.where(labels == cluster)[0]
    ix1 = idx[np.argmin(np.sum(distances[idx][:, idx], axis=1))]
    return ix1

def get_pair(coordinates, distances, labels, cluster1, cluster2, kind='medoids'):
    if kind == 'medoids':
        ix1 = get_medoids(distances, labels, cluster1)            
        ix2 = get_medoids(distances, labels, cluster2)
    elif kind == 'timewise':
#         get the closes point from cluster1
        idx1 = np.where(labels == cluster1)[0]
        ix1 = idx1[np.argmin(np.sum(coordinates[idx1, :]**2, axis=1))]        
        idx2 = np.where(labels == cluster2)[0]
        ix2 = idx2[np.argmin(np.sum(coordinates[idx2, :]**2, axis=1))]
#         idx2 = np.where(labels == cluster2)[0]
#         ix2 = idx2[np.argmin(distances[ix1][idx2])]
    
    x1 = coordinates[ix1]
    x2 = coordinates[ix2]

    print(f"Average shortest distance from {cluster1} to {cluster2}: {distances[ix1][ix2]:.2f}")
    return x1, x2


def get_interpolated_coordinates(coordinates, labels, clusters, distances, 
                                 clusters_from, clusters_to, 
                                 points_list = None,
                                 n_points=10, n_starts=10, space='poincare', seed=42):    
    
    interpolated_coordinates = np.copy(coordinates)
    interpolated_labels = np.copy(labels)
    
    if points_list is None:
        points_list = []
        for i1, cl_from in enumerate(clusters_from):
            for i2, cl_to in enumerate(clusters_to):
                if cl_from != cl_to:
                    print(f"{cl_from} to {cl_to}")
                    for i in range(n_starts):
                        u, v, iu, iv = get_closest(coordinates, distances, clusters, cluster_from=cl_from, cluster_to=cl_to, seed=seed)
                        points_list.append([iu, iv])
    
    for p_pair in points_list:        
        u = coordinates[p_pair[0]]
        v = coordinates[p_pair[1]]
        if space == 'poincare':
            interpolated_coordinates = np.concatenate((interpolated_coordinates, 
                                                       poincare_linspace(u, v, n_points=n_points)))
        else:            
            interpolated_coordinates = np.concatenate((interpolated_coordinates, 
                                                       intermediates(u, v, nb_points=n_points)))

        interpolated_labels = np.concatenate((np.array(interpolated_labels), 
                          np.array(['interpolation']*n_points)))
                
    return interpolated_coordinates, interpolated_labels, points_list


def get_time_and_idx(dpt, idx):
    time = dpt[idx] / np.max(dpt[idx])
    ix_time = np.argsort(time)

    return time, ix_time


def plot_dtw_comparison(dpt_true, dpt_po, dpt_fa, dpt_umap, 
    idx_full, idx_pm, idx_fa, idx_umap, data_full, 
    x_predicted_po, x_predicted_fa, x_predicted_umap, 
    col_names, fout,
    n_plt = 15, n2 = 4, win = 5, pl_size = 2, fs = 9,
                        lw = 2, cpal = ['#1a9641', '#d7191c', '#2b83ba', '#fdae61']):

    time_true, ix_true = get_time_and_idx(dpt_true, idx_full)
    time_po, ix_po = get_time_and_idx(dpt_po, idx_pm)
    time_fa, ix_fa = get_time_and_idx(dpt_fa, idx_fa)
    time_umap, ix_umap = get_time_and_idx(dpt_umap, idx_umap)

    N = len(col_names)

    n1 = n_plt // n2
    if n1*n2 < n_plt:
        n1 += 1

    if n1 == 1:
        n1 = 2
    if n2 == 1:
        n2 = 2
        
    n1_all = N // n2
    if n2*n1_all < N:
        n1_all += 1

    if n1 == 1:
        n1 = 2
    if n2 == 1:
        n2 = 2
    

    fig, axs = plt.subplots(n1, n2, sharey=False, figsize=(n2*pl_size + 2, n1*pl_size))

    i = 0
    dtw_po = []
    dtw_fa = []
    dtw_umap = []
    for i1 in range(n1_all):
        for i2 in range(n2):            
            if i < N:
                df_true = pd.DataFrame(data_full[idx_full[ix_true], i], columns=['gene'])
                y_smooth_true = df_true.rolling(window=win, min_periods=1).mean()['gene'].values

                df_po = pd.DataFrame(x_predicted_po[idx_pm[ix_po], i], columns=['gene'])
                y_smooth_po = df_po.rolling(window=win, min_periods=1).mean()['gene'].values

                df_fa= pd.DataFrame(x_predicted_fa[idx_fa[ix_fa], i], columns=['gene'])
                y_smooth_fa = df_fa.rolling(window=win, min_periods=1).mean()['gene'].values

                df_umap= pd.DataFrame(x_predicted_umap[idx_umap[ix_umap], i], columns=['gene'])
                y_smooth_umap = df_umap.rolling(window=win, min_periods=1).mean()['gene'].values

                distance, path = fastdtw(y_smooth_true, y_smooth_po, dist=euclidean)
                dtw_po.append(distance)
                    
                distance, path = fastdtw(y_smooth_true, y_smooth_fa, dist=euclidean)
                dtw_fa.append(distance)

                distance, path = fastdtw(y_smooth_true, y_smooth_umap, dist=euclidean)
                dtw_umap.append(distance)

                if i < n_plt:
                    
                    axs[i1, i2].grid('off')
                    axs[i1, i2].yaxis.set_tick_params(labelsize=fs)
                    axs[i1, i2].xaxis.set_tick_params(labelsize=fs)
                    marker = col_names[i]
                    axs[i1, i2].plot(time_true[ix_true], y_smooth_true, c=cpal[0], linewidth=lw*2)  
                    axs[i1, i2].plot(time_po[ix_po], y_smooth_po,  c=cpal[1], linewidth=lw)                    
                    axs[i1, i2].plot(time_fa[ix_fa], y_smooth_fa,  c=cpal[2], linewidth=lw)
                    axs[i1, i2].plot(time_umap[ix_umap], y_smooth_umap,  c=cpal[3], linewidth=lw)

                    axs[i1, i2].set_title(marker, fontsize=fs)            

                elif i < n1*n2:
                # else:
                    axs[i1, i2].axis('off')
                    axs[i1, i2].grid('off')
                    axs[i1, i2].yaxis.set_tick_params(labelsize=fs)
                    axs[i1, i2].xaxis.set_tick_params(labelsize=fs)

                if i == (n_plt-1):
                    i1l = i1
                    i2l = i2
            elif i < n1*n2:
                axs[i1, i2].axis('off')
                axs[i1, i2].grid('off')
                axs[i1, i2].yaxis.set_tick_params(labelsize=fs)
                axs[i1, i2].xaxis.set_tick_params(labelsize=fs)

                
            i+=1

    dtw_po = np.array(dtw_po)
    dtw_fa = np.array(dtw_fa)
    dtw_umap = np.array(dtw_umap)

    axs[i1l, i2l].legend(['True', f'PM: {np.median(dtw_po):.1f}', f'FA2: {np.median(dtw_fa):.1f}', f'UMAP: {np.median(dtw_umap):.1f}'], 
                     bbox_to_anchor=(1.4, 0.5), fontsize=fs)
    # axs[i1l, i2l].legend(['True', f'Poincaré', f'ForceAtals2', f'UMAP'], 
    #                  bbox_to_anchor=(1.4, 0.5), fontsize=fs)
                     

    plt.xlabel('pseudotime', fontsize=fs)
    fig.tight_layout()

    plt.savefig(fout + '_compare_interpolation.pdf', format='pdf')

    return dtw_po, dtw_fa, dtw_umap


def plot_benchmark(coord, labels, method, fout, d1=2.5, d2=2.5, fs=9, ms=3):
    sns.set_style("white")
    print(np.shape(coord))
    fig = plt.figure(figsize=(d1, d2))
    ax = plt.gca()
    axs_names=['x1', 'x2']
    idx = np.random.permutation(len(coord))
    df = pd.DataFrame(coord[idx, :], columns=axs_names)
    df['labels'] = labels[idx]
    # plt.title(f'Interpolation: {method}', fontsize=fs)
    ax.axis('equal')
    ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])
    sns.scatterplot(x=axs_names[0], y=axs_names[1], hue='labels',
                    alpha=1.0, edgecolor="none",
                    palette=None,
                    data=df, ax=ax, s=ms)
    ax.set_xlabel(axs_names[0], fontsize=fs)
    ax.set_ylabel(axs_names[1], fontsize=fs)
    ax.legend_.remove()
    fig.tight_layout()
    plt.savefig( f"{fout}_{method}_interpolation.pdf", format='pdf')
