"""this file tries to replicate https://github.com/koraykv/unsup/blob/master/demo/demo_fista.lua,
with its original experiment parameters.
the original script's data for first 10 iterations were collected using `/debug/reference/demo_fista_debug.lua`


run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python demo_fista_custom.py` to get full reproducibility.
https://discuss.pytorch.org/t/nondeterministic-behaviour-of-grad-running-on-cpu/7402/2
"""
import os
from sys import argv
import numpy as np
from numpy.linalg import norm
from unsup import dir_dictionary, sc  # somehow I need to import spams earlier than torch.
from torch import Tensor, optim
from torch.autograd import Variable


def demo(data_dir=dir_dictionary['debug_reference'],
         data_file_prefix='demo_fista_conv_debug', lam=0.1, gpu=False):
    # load all data.
    data_dict = {
        k: np.load(os.path.join(data_dir, f'{data_file_prefix}_{k}.npy')) for k in ('data', 'serr',
                                                                                    'weight', 'grad_weight')
    }
    # print(data_dict['weight'].shape)
    num_data = data_dict['data'].shape[0]
    assert num_data == data_dict['serr'].shape[0] == data_dict['weight'].shape[0] == data_dict['grad_weight'].shape[0]

    model = sc.ConvSC(4, 9, lam, im_size=25, legacy_bias=True)
    # intialize.
    if gpu:
        model.cuda()

    # init_weight = Tensor(np.ascontiguousarray(data_dict['weight'][0][:, np.newaxis,::-1,::-1]))
    init_weight = Tensor(np.ascontiguousarray(data_dict['weight'][0][:, np.newaxis]))
    assert model.linear_module.weight.size() == init_weight.size()
    if gpu:
        init_weight = init_weight.cuda()
    model.linear_module.weight.data[...] = init_weight
    # TODO: fix this later.
    optimizer = optim.SGD(model.parameters(), lr=0.01 / 4)

    # let's have the for loop
    for i, (data_this, weight_this, grad_this, serr_this) in enumerate(
            zip(data_dict['data'], data_dict['weight'], data_dict['grad_weight'], data_dict['serr'])
    ):
        # if i >= 5:
        #     break
        assert data_this.shape == (1, 25, 25)
        assert weight_this.shape == (4, 9, 9) == grad_this.shape
        assert np.isscalar(serr_this)
        print(i)
        # ok. let's compute cost.
        # new_weight = Tensor(weight_this[:, np.newaxis])
        # if gpu:
        #     new_weight = new_weight.cuda()
        # model.linear_module.weight.data[...] = new_weight

        weight_this_actual = model.linear_module.weight.data.cpu().numpy()
        assert weight_this_actual.shape == (4, 1, 9, 9)
        print(norm(weight_this_actual.ravel() - weight_this.ravel()) / (norm(weight_this.ravel()) + 1e-6))

        optimizer.zero_grad()
        input_this = Tensor(data_this[np.newaxis])
        if gpu:
            input_this = input_this.cuda()
        cost_this, _ = model.forward(Variable(input_this))
        cost_this.backward()
        cost_this = cost_this.data.cpu().numpy()
        assert cost_this.shape == (1,)
        cost_this = cost_this[0]
        print(cost_this, serr_this)

        # check grad
        grad_this_actual = model.linear_module.weight.grad.data.cpu().numpy()
        assert grad_this_actual.shape == (4, 1, 9, 9)  # （4,1,9,9) and (4,9,9) are compatible for ravel().
        print(norm(grad_this_actual.ravel() - grad_this.ravel()) / (norm(grad_this.ravel()) + 1e-6))

        optimizer.step()


if __name__ == '__main__':
    gpu = len(argv) > 1
    if gpu:
        print('GPU support!')
    demo(gpu=gpu)
    # sample output on Mac (CPU, openblas + pytorch pip, no flags like OMP_NUM_THREADS=1 MKL_NUM_THREADS=1,
    # I wonder if MKL won't work fully on Mac)
    #
    # 0
    # 2.64262845059e-08
    # 13.856 13.856
    # 1.23318449223e-06
    # 1
    # 3.94763210149e-08
    # 3.87481 3.87481
    # 6.22817354576e-07
    # 2
    # 4.69356241154e-08
    # 2.66957 2.66957
    # 3.38883078642e-07
    # 3
    # 5.16161843593e-08
    # 6.69115 6.69115
    # 1.09500513156e-06
    # 4
    # 5.83944909115e-08
    # 2.12411 2.12411
    # 3.56212329169e-07
    # 5
    # 6.56050810036e-08
    # 8.3021 8.30209
    # 1.02457296263e-06
    # 6
    # 6.89768787876e-08
    # 7.59268 7.59268
    # 5.41386326617e-07
    # 7
    # 7.16579647536e-08
    # 7.99279 7.99279
    # 1.07059139945e-06
    # 8
    # 7.33847423445e-08
    # 14.9272 14.9272
    # 7.81217008352e-07
    # 9
    # 8.13017700279e-08
    # 1.22348 1.22348
    # 2.12032878927e-07
    #
    # GPU
    #
    # GPU support!
    # 0
    # 2.64262845059e-08
    # 13.856 13.856
    # 1.10187580374e-06
    # 1
    # 3.91455569019e-08
    # 3.87481 3.87481
    # 6.15332941879e-07
    # 2
    # 4.57623152252e-08
    # 2.66957 2.66957
    # 3.33932567063e-07
    # 3
    # 5.01966977462e-08
    # 6.69116 6.69115
    # 1.08981075201e-06
    # 4
    # 5.58040654146e-08
    # 2.12411 2.12411
    # 3.72088373824e-07
    # 5
    # 6.31944666596e-08
    # 8.30209 8.30209
    # 9.90504647911e-07
    # 6
    # 6.93648052698e-08
    # 7.59268 7.59268
    # 6.80754755274e-07
    # 7
    # 7.21239462772e-08
    # 7.99279 7.99279
    # 9.40726707946e-07
    # 8
    # 7.61989855395e-08
    # 14.9272 14.9272
    # 7.03189175642e-07
    # 9
    # 8.2560026512e-08
    # 1.22348 1.22348
    # 2.98343133683e-07
