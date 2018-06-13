"""this file tries to replicate https://github.com/koraykv/unsup/blob/master/demo/demo_fista.lua,
with its original experiment parameters.
the original script's data for first 10 iterations were collected using `/debug/reference/demo_fista_debug.lua`


run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python demo_fista_custom.py` to get full reproducibility.
https://discuss.pytorch.org/t/nondeterministic-behaviour-of-grad-running-on-cpu/7402/2
"""
import os
from sys import argv
import numpy as np
from torch import Tensor, optim
from torch.autograd import Variable
import torch
from torch.utils.data import TensorDataset, DataLoader
import h5py
from unsup import sc  # somehow I need to import spams earlier than torch.


# use argparse to handle

def demo(lam, gpu=True, seed=0, batch_size=16):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # load all data.
    with h5py.File(os.path.join(os.path.split(__file__)[0], 'data.hdf5'), 'r') as f_input:
        data_raw = f_input['data'][...]
    assert data_raw.shape[0] == 1000000

    # for 0.3.1, we need dummy label tensor
    # TODO: fix this in 0.4.0
    data_tensor = TensorDataset(Tensor(data_raw), Tensor(np.zeros(data_raw.shape[0],
                                                                  dtype=np.int64)))
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True, drop_last=False)
    num_iter_per_epoch = (data_raw.shape[0] + batch_size - 1) // batch_size

    model = sc.ConvSC(64, 9, lam, im_size=25)
    # whatever std should be fine, as normalization will happen later.
    assert model.linear_module.bias is None
    model.linear_module.weight.data.normal_(0, 1)
    model.normalize()
    # intialize.
    if gpu:
        model.cuda()

    # no momentum, as I do normalization.
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    epoch_num = 0
    while True:
        loss_sum = 0.0
        for i_minibatch, (input_this, _) in enumerate(data_loader):
            if gpu:
                # TODO: here, I don't need to handle the issue of train vs eval mode
                # as I don't have such modules.
                input_this = Variable(input_this.cuda())
            else:
                input_this = Variable(input_this)
            optimizer.zero_grad()

            # get input from dataloader.

            cost_this, _ = model.forward(input_this)
            # so that gradient magnitude won't scale with batch size.
            cost_this = cost_this / input_this.size()[0]
            cost_this.backward()
            cost_this = cost_this.data.cpu().numpy()
            assert cost_this.shape == (1,)
            cost_this = cost_this[0]
            loss_sum += cost_this

            optimizer.step()
            model.normalize()

            if (i_minibatch + 1) % 100 == 0:
                print(f'epoch {epoch_num}, iter {i_minibatch+1}/{num_iter_per_epoch}', loss_sum)
                loss_sum = 0.0

        epoch_num += 1

        # save images.
        # follow the old practice.
        # later on.


if __name__ == '__main__':
    demo(1.0, batch_size=16)
