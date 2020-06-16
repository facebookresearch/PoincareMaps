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


def train(model, data, optimizer, args, fout=None, labels=None, earlystop=0.0, color_dict=None):
    loader = DataLoader(data, batch_size=args.batchsize, shuffle=True)

    pbar = tqdm(range(args.epochs), ncols=80)

    n_iter = 0
    epoch_loss = []
    t_start = timeit.default_timer()
    earlystop_count = 0
    for epoch in pbar:        
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
            
            grad_norm.append(model.lt.weight.grad.data.norm().item())            

            n_iter += 1

        epoch_error /= len(loader)
        epoch_loss.append(epoch_error)
        pbar.set_description("loss: {:.5f}".format(epoch_error))

        if epoch > 10:
            delta = abs(epoch_loss[epoch] - epoch_loss[epoch-1])            
            if (delta < earlystop):                
                earlystop_count += 1
            if earlystop_count > 50:
                print(f'\nStopped at epoch {epoch}')
                break

        if args.debugplot:
            if (epoch % args.debugplot) == 0:
                d = model.lt.weight.cpu().detach().numpy()
                titlename = 'epoch: {:d}, loss: {:.3e}'.format(
                    epoch, np.mean(epoch_loss))

                if epoch > 5:
                    plotPoincareDisc(np.transpose(d), labels, fout, titlename, color_dict=color_dict)
                    np.savetxt(fout + '.csv', d, delimiter=",")

                ball_norm = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
                if np.max(ball_norm) > 1.001:
                    print('The learning rate is too high.')

                delta = abs(epoch_loss[epoch] - epoch_loss[epoch-1])
                plot_training(epoch_loss, 
                title_name=f'd={delta:.2e}', 
                file_name=fout+'_loss', d1=4, d2=4)

                # print(f"{epoch}: time={elapsed:.3f}, "
                #       f"loss = {np.mean(epoch_loss):.3e}, "
                #       f"grad_norm = {np.mean(grad_norm):.3e}, "
                #       f"max_norm = {np.max(ball_norm):.4f}, "
                #       f"mean_norm = {np.mean(ball_norm):.4f}")


    print(f"PM computed in {(timeit.default_timer() - t_start):.2f} sec")

    delta = abs(epoch_loss[epoch] - epoch_loss[epoch-1])
    plot_training(epoch_loss, title_name=f'd={delta:.2e}', file_name=fout+'_loss', d1=4, d2=4)

    return model.lt.weight.cpu().detach().numpy(), epoch_error, epoch
