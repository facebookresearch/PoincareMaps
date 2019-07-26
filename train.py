# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import timeit
from torch.utils.data import DataLoader
from visualize import *
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter


def train(model, data, optimizer, args, fout=None, labels=None, tb=0):
    loader = DataLoader(data, batch_size=args.batchsize, shuffle=True)

    if tb:
        print('Start TensorBoard')
        writer = SummaryWriter()

    pbar = tqdm(range(args.epochs), ncols=80)

    n_iter = 0
    for epoch in pbar:
        if args.debugplot:
            epoch_loss = []            
            t_start = timeit.default_timer()

        grad_norm = []

        # determine learning rate
        lr = args.lr
        if epoch < args.burnin:
            lr = lr * args.lrm

        epoch_error = 0
        for inputs, targets in loader:
            loss = model.lossfn(model(inputs), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lr=lr)

            epoch_error += loss.item()

            if args.debugplot:
                elapsed = timeit.default_timer() - t_start
            
            grad_norm.append(model.lt.weight.grad.data.norm().item())

            if tb:
                writer.add_scalar("data/train/error", loss.item(), n_iter)
                writer.add_scalar("data/train/gradients", grad_norm[-1], n_iter)                

            n_iter += 1


        epoch_error /= len(loader)

        if tb:
            writer.add_scalar("data/train/epoch_error", epoch_error, epoch)

        pbar.set_description("loss: {:.5f}".format(epoch_error))

        if args.debugplot:
            if (epoch % args.debugplot) == 0:
                d = model.lt.weight.data.numpy()
                titlename = 'epoch: {:d}, loss: {:.3e}'.format(
                    epoch, np.mean(epoch_loss))

                if epoch > 5:
                    plotPoincareDisc(np.transpose(d), labels, fout, titlename)
                    np.savetxt(fout + '.csv', d, delimiter=",")

                ball_norm = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
                if np.max(ball_norm) > 1.001:
                    print('The learning rate is too high.')

                print(f"{epoch}: time={elapsed:.3f}, "
                      f"loss = {np.mean(epoch_loss):.3e}, "
                      f"grad_norm = {np.mean(grad_norm):.3e}, "
                      f"max_norm = {np.max(ball_norm):.4f}, "
                      f"mean_norm = {np.mean(ball_norm):.4f}")

    if tb:
        writer.close()

    return model.lt.weight.cpu().detach().numpy(), epoch_error
