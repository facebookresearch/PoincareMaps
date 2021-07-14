# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.metrics.pairwise import pairwise_distances
from torch.autograd import Function
from torch import nn
import numpy as np
import torch

eps = 1e-5
boundary = 1 - eps

def poincare_translation(v, x):
    """
    Computes the translation of x  when we move v to the center.
    Hence, the translation of u with -u should be the origin.
    """
    xsq = (x ** 2).sum(axis=1)
    vsq = (v ** 2).sum()
    xv = (x * v).sum(axis=1)
    a = np.matmul((xsq + 2 * xv + 1).reshape(-1, 1),
                  v.reshape(1, -1)) + (1 - vsq) * x
    b = xsq * vsq + 2 * xv + 1
    return np.dot(np.diag(1. / b), a)


def poincare_root(root_name, labels, features):
    if root_name is not None:
        head_idx = np.where(labels == root_name)[0]

        if len(head_idx) > 1:
            # medoids in Euclidean space
            D = pairwise_distances(features[head_idx, :], metric='euclidean')
            return head_idx[np.argmin(D.mean(axis=0))]
        elif len(head_idx) == 1:
            return head_idx[0]
        else:
            return -1

    return -1


def grad(x, v, sqnormx, sqnormv, sqdist):
    alpha = (1 - sqnormx)
    beta = (1 - sqnormv)        
    z = 1 + 2 * sqdist / (alpha * beta)
    a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) /
            torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
    a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
    z = torch.sqrt(torch.pow(z, 2) - 1)
    z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
    return 4 * a / z.expand_as(x)


class PoincareDistance(Function):
    @staticmethod
    def forward(self, u, v):  
        self.save_for_backward(u, v)
        self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
        self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
        self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(self, g):    
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

    
def klSym(preds, targets):
    # preds = preds + eps
    # targets = targets + eps
    logPreds = preds.clamp(1e-20).log()
    logTargets = targets.clamp(1e-20).log()
    diff = targets - preds
    return (logTargets * diff - logPreds * diff).sum() / len(preds)


class PoincareEmbedding(nn.Module):
    def __init__(self,
                 size,
                 dim,
                 dist=PoincareDistance,
                 max_norm=1,
                 Qdist='laplace',
                 lossfn='klSym',
                 gamma=1.0,
                 cuda=0):
        super(PoincareEmbedding, self).__init__()

        self.dim = dim
        self.size = size
        self.lt = nn.Embedding(size, dim, max_norm=max_norm)
        self.lt.weight.data.uniform_(-1e-4, 1e-4)
        self.dist = dist
        self.Qdist = Qdist
        self.lossfnname = lossfn
        self.gamma = gamma

        self.sm = nn.Softmax(dim=1)
        self.lsm = nn.LogSoftmax(dim=1)

        if lossfn == 'kl':
            self.lossfn = nn.KLDivLoss()
        elif lossfn == 'klSym':
            self.lossfn = klSym
        elif lossfn == 'mse':
            self.lossfn = nn.MSELoss()
        else:
            raise NotImplementedError

        if cuda:
            self.lt.cuda()

    def forward(self, inputs):
        embs_all = self.lt.weight.unsqueeze(0)
        embs_all = embs_all.expand(len(inputs), self.size, self.dim)

        embs_inputs = self.lt(inputs).unsqueeze(1)
        embs_inputs = embs_inputs.expand_as(embs_all)

        dists = self.dist().apply(embs_inputs, embs_all).squeeze(-1)        

        if self.lossfnname == 'kl':
            if self.Qdist == 'laplace':
                return self.lsm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.lsm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.lsm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == 'klSym':
            if self.Qdist == 'laplace':
                return self.sm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.sm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.sm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == 'mse':
            return self.sm(-self.gamma * dists)
        else:
            raise NotImplementedError
