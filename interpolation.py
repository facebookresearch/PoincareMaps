# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
fs=9
lw = 2
cpal = ['#4daf4a', '#e41a1c', '#377eb8', '#abd9e9']

def get_pair(coordinates, distances, labels, cluster1, cluster2):    
    ix1 = np.where(labels == cluster1)[0][0]
    x1 = coordinates[ix1]    
    idx = np.where(labels == cluster2)[0]
    ix2 = idx[np.argmin(distances[ix1][idx])]
    x2 = coordinates[ix2]
    return x1, x2


def get_geodesic_parameters(u, v):
    nu = u[0]**2 + u[1]**2
    nv = v[0]**2 + v[1]**2
    a = (u[1]*nv - v[1]*nu + u[1] - v[1]) / (u[0]*v[1] - u[1]*v[0])
    b = (v[0]*nu - u[0]*nv + v[0] - u[0]) / (u[0]*v[1] - u[1]*v[0])
    return a, b
    
    
def poincare_linspace(a, b, u, v, num=75, space='lin'):
#     start = min(u[0], v[0])
#     fin = max(u[0], v[0])
    if space == 'lin':
        x = np.linspace(u[0], v[0], num=num)
    else:
        x = np.exp(np.linspace(log1p(u[0]), log1p(v[0]), num=num)) - 1
    
    interpolated = np.zeros([len(x), 2])
    interpolated[:, 0] =x 
    
    r = a**2/4 + b**2/4 - 1
    y_1 = -b/2 + np.sqrt(r - (x+a/2)**2)
    y_2 = -b/2 - np.sqrt(r - (x+a/2)**2)
    
    if max(x**2 + y_1**2) > 1:
        interpolated[:, 1] = y_2 
    else:
        interpolated[:, 1] = y_1
    return interpolated


def get_interpolated_coordinates(coordinates, labels, clusters, distances, 
                                 clusters_from, clusters_to, 
                                 points_list = None,
                                 n_points=10, n_starts=10, space='poincare'):
    
    interpolated_coordinates = np.copy(coordinates)
    interpolated_labels = np.copy(labels)
    
    if points_list is None:
        points_list = []
        for i1, cl_from in enumerate(clusters_from):
            for i2, cl_to in enumerate(clusters_to):
                if cl_from != cl_to:
                    print(f"{cl_from} to {cl_to}")
                    for i in range(n_starts):
                        u, v, iu, iv = get_closest(coordinates, distances, clusters, cluster_from=cl_from, cluster_to=cl_to)
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
    idx_full, idx_po, idx_fa, idx_umap, data_full, 
    x_predicted_po, x_predicted_fa, x_predicted_umap, 
    col_names, fout,
    n_plt = 15, n2 = 3, win = 5, pl_size = 2):

    time_true, ix_true = get_time_and_idx(dpt_true, idx_full)
    time_po, ix_po = get_time_and_idx(dpt_po, idx_po)
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
    

    fig, axs = plt.subplots(n1, n2, sharey=False, figsize=(n2*pl_size + 2, n1*pl_size))

    i = 0
    dtw_po = []
    dtw_fa = []
    dtw_umap = []
    for i1 in range(n1):
        for i2 in range(n2):
            axs[i1, i2].grid('off')
            axs[i1, i2].yaxis.set_tick_params(labelsize=fs)
            axs[i1, i2].xaxis.set_tick_params(labelsize=fs)
            if i < N:
                df_true = pd.DataFrame(data_full[idx_full[ix_true], i], columns=['gene'])
                y_smooth_true = df_true.rolling(window=win, min_periods=1).mean()['gene'].values

                df_po = pd.DataFrame(x_predicted_po[idx_pm[ix_po], i], columns=['gene'])
                y_smooth_po = df_po.rolling(window=win, min_periods=1).mean()['gene'].values

                df _fa= pd.DataFrame(x_predicted_fa[idx_bm[ix_fa], i], columns=['gene'])
                y_smooth_fa = df_fa.rolling(window=win, min_periods=1).mean()['gene'].values

                df_umap= pd.DataFrame(x_predicted_umap[idx_bm[ix_fa], i], columns=['gene'])
                y_smooth_umap = df_umap.rolling(window=win, min_periods=1).mean()['gene'].values

                distance, path = fastdtw(y_smooth_true, y_smooth_po, dist=euclidean)
                dtw_po.append(distance)
                    
                distance, path = fastdtw(y_smooth_true, y_smooth_fa, dist=euclidean)
                dtw_fa.append(distance)

                distance, path = fastdtw(y_smooth_true, y_smooth_umap, dist=euclidean)
                dtw_umap.append(distance)

                if i < n_plt:
                    marker = col_names[i]
                    axs[i1, i2].plot(time_true[ix_true], y_smooth_true, c=cpal[0], linewidth=lw*2)  
                    axs[i1, i2].plot(time_po[ix_po], y_smooth_po, c=cpal[1], linewidth=lw)                    
                    axs[i1, i2].plot(time_fa[ix_fa], y_smooth_fa, c=cpal[2], linewidth=lw)
                    axs[i1, i2].plot(time_umap[ix_fa], y_smooth_umap, c=cpal[3], linewidth=lw)

                    axs[i1, i2].set_title(marker, fontsize=fs)            
                else:
                    axs[i1, i2].axis('off')
                
            i+=1

    dtw_po = np.array(dtw_po)
    dtw_fa = np.array(dtw_fa)
    dtw_umap = np.array(dtw_umap)

    axs[i1, i2].legend(['True', f'Poincaré: {np.median(dtw_po):.2f}', f'ForceAtals2: {np.median(dtw_fa):.2f}', f'UMAP: {np.median(dtw_umap):.2f}'], 
                     bbox_to_anchor=(1.4, 0.5), fontsize=fs)
                     
    # axs[3, 1].legend(['True', f'Poincaré: {dtw_po:.2f}', f'{method}: {dtw_bm:.2f}'], 
    #                  loc='center left', bbox_to_anchor=(1.4, 0.5), fontsize=fs)

    plt.xlabel('pseudotime', fontsize=fs)
    fig.tight_layout()

    plt.savefig(fout + '_compare_interpolation.pdf', format='pdf')

    return dtw_po, dtw_fa, dtw_umap
