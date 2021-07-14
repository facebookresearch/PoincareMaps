# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import random

warnings.filterwarnings("ignore")
plt.switch_backend('agg')
sns.set()


colors_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                  '#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B',
                  '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                  '#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B',
                  '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
                  '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
                  '#CCEBC5', '#FFED6F', '#edf8b1', '#c7e9b4', '#7fcdbb',
                  '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']



def linear_scale(embeddings):
    embeddings = np.transpose(embeddings)
    sqnorm = np.sum(embeddings ** 2, axis=1, keepdims=True)
    dist = np.arccosh(1 + 2 * sqnorm / (1 - sqnorm))
    dist = np.sqrt(dist)
    dist /= dist.max()
    sqnorm[sqnorm == 0] = 1
    embeddings = dist * embeddings / np.sqrt(sqnorm)
    return np.transpose(embeddings)

def plot_training(loss_func, title_name=None, file_name=None, d1=4, d2=4, fs=11):
  fig = plt.figure(figsize=(d1, d2))
  plt.plot(loss_func, c='#f03b20')

  if title_name:
    plt.title(title_name, fontsize=fs)
  plt.show()

  if file_name:
    plt.savefig(file_name + '.png', format='png')
  plt.close(fig)



def plot_poincare_disc(x, labels=None, labels_name='labels', labels_order=None, 
                       file_name=None, coldict=None,
                       d1=19, d2=18.0, fs=11, ms=20, col_palette=plt.get_cmap("tab10"), bbox=(1.3, 0.7)):    

    idx = np.random.permutation(len(x))
    df = pd.DataFrame(x[idx, :], columns=['pm1', 'pm2'])
    
    fig = plt.figure(figsize=(d1, d2))
    ax = plt.gca()
    circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)

    if not (labels is None):
        df[labels_name] = labels[idx]
        if labels_order is None:
            labels_order = np.unique(labels)        
        if coldict is None:
            coldict = dict(zip(labels_order, col_palette[:len(labels)]))
        sns.scatterplot(x="pm1", y="pm2", hue=labels_name, 
                        hue_order=labels_order,
                        palette=coldict,
                        alpha=1.0, edgecolor="none",
                        data=df, ax=ax, s=ms)

        ax.legend(fontsize=fs, loc='best', bbox_to_anchor=bbox)
            
    else:
        sns.scatterplot(x="pm1", y="pm2",
                        data=df, ax=ax2, s=ms)
    fig.tight_layout()
    ax.axis('off')
    ax.axis('equal')  

    labels_list = np.unique(labels)
    for l in labels_list:
#         i = np.random.choice(np.where(labels == l)[0])
        ix_l = np.where(labels == l)[0]
        c1 = np.median(x[ix_l, 0])
        c2 = np.median(x[ix_l, 1])
        ax.text(c1, c2, l, fontsize=fs)


    if file_name:
        plt.savefig(file_name + '.png', format='png')

    plt.close(fig)


def plotPoincareDisc(x,
                     label_names=None,
                     file_name=None,
                     title_name=None,
                     idx_zoom=None,
                     show=False,
                     d1=12,
                     d2=6,
                     fs=11,
                     ms=4,
                     col_palette=None,
                     color_dict=None):
    if col_palette is None:
        col_palette = colors_palette
        # col_palette = plt.get_cmap("tab10")

    df = pd.DataFrame(dict(x=x[0], y=x[1], label=label_names))
    groups = df.groupby('label')

    fig = plt.figure(figsize=(d1, d2), dpi=300)
    circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')

    plt.subplot(1, 2, 1)
    plt.gca().add_patch(circle)
    plt.plot(0, 0, 'x', c=(0, 0, 0), ms=ms)
    plt.title(title_name, fontsize=fs)

    if color_dict is None:
        j = 0
        color_dict = {}
        for name, group in groups:
            color_dict[name] = col_palette[j]
            j += 1

    marker = 'o'
    for name, group in groups:        
        plt.plot(group.x, group.y, marker=marker, markerfacecolor='none',
                 c=color_dict[name], linestyle='', ms=ms, label=name)
    plt.plot(0, 0, 'x', c=(1, 1, 1), ms=ms)
    plt.axis('off')
    plt.axis('equal')
    # plt.legend(numpoints=1, loc='center left',
    #            bbox_to_anchor=(1, 0.5), fontsize=fs)

    labels_list = np.unique(label_names)

    for l in labels_list:
#         i = np.random.choice(np.where(labels == l)[0])
        ix_l = np.where(label_names == l)[0]
        c1 = np.median(x[0, ix_l])
        c2 = np.median(x[1, ix_l])
        plt.text(c1, c2, l, fontsize=fs)
#
    if idx_zoom is None:
        xl = np.array(linear_scale(x))
        xl[np.isnan(xl)] = 0

        df = pd.DataFrame(dict(x=xl[0], y=xl[1], label=label_names))
        groups = df.groupby('label')
    else:
        xl = np.array(linear_scale(x[:, idx_zoom]))
        xl[np.isnan(xl)] = 0

        df = pd.DataFrame(dict(x=xl[0], y=xl[1], label=label_names[idx_zoom]))
        groups = df.groupby('label')

    circle = plt.Circle((0, 0), radius=1, fc='none',
                        color='black', linestyle=':')
    plt.subplot(1, 2, 2)
    plt.gca().add_patch(circle)
    plt.plot(0, 0, 'x', c=(0, 0, 0), ms=ms)
    plt.title('zoom in', fontsize=fs)

    for name, group in groups:
        plt.plot(group.x, group.y, marker=marker, markerfacecolor='none',
                 c=color_dict[name], linestyle='', ms=ms, label=name)

    plt.plot(0, 0, 'x', c=(1, 1, 1), ms=6)

    plt.axis('off')
    plt.axis('equal')

    plt.legend(numpoints=1, loc='center left',
               bbox_to_anchor=(1, 0.5), fontsize=fs)

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name + '.png', format='png')

    if show:
        plt.show()

    plt.close(fig)

    return color_dict


def plot2D(x,
           label_names=None,
           file_name=None,
           title_name=None,
           idx_zoom=None,
           show=False,
           d1=7,
           d2=7,
           fs=8,
           ms=4,
           col_palette=None):
    if col_palette is None:
        col_palette = colors_palette

    df = pd.DataFrame(dict(x=x[0], y=x[1], label=label_names))
    groups = df.groupby('label')

    fig = plt.figure(figsize=(d1, d2), dpi=300)
    plt.title(title_name, fontsize=fs)

    j = 0
    color_dict = {}
    for name, group in groups:
        marker = 'o'
        if name in set(['Ery', 'Mk', 'MEP']):
            marker = 'v'
        elif name == 'Lymph':
            marker = 's'
        plt.plot(group.x, group.y, marker=marker, markerfacecolor='none',
                 c=col_palette[j], linestyle='', ms=ms, label=name)
        color_dict[name] = col_palette[j]
        j += 1

    plt.legend(numpoints=1, loc='center left',
               bbox_to_anchor=(1, 0.5), fontsize=fs)

    if file_name:
        plt.savefig(file_name + '.png', format='png')

    if show:
        plt.show()

    plt.close(fig)
