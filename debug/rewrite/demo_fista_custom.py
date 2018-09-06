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
         data_file_prefix='demo_fista_debug', lam=1.0, gpu=False):
    # load all data.
    data_dict = {
        k: np.load(os.path.join(data_dir, f'{data_file_prefix}_{k}.npy')) for k in ('data', 'serr',
                                                                                    'weight', 'grad_weight')
    }
    # print(data_dict['weight'].shape)
    num_data = data_dict['data'].shape[0]
    assert num_data == data_dict['serr'].shape[0] == data_dict['weight'].shape[0] == data_dict['grad_weight'].shape[0]

    model = sc.LinearSC(81, 32, lam, solver_type='fista_custom')
    # intialize.
    if gpu:
        model.cuda()

    init_weight = Tensor(data_dict['weight'][0])
    assert model.linear_module.weight.size() == init_weight.size()
    if gpu:
        init_weight = init_weight.cuda()
    model.linear_module.weight.data[...] = init_weight
    # change factor from 32 to 81 or 1 will dramatically increase the difference between ref and actual.
    optimizer = optim.SGD(model.parameters(), lr=0.01 / 32)

    # let's have the for loop
    for i, (data_this, weight_this, grad_this, serr_this) in enumerate(
            zip(data_dict['data'], data_dict['weight'], data_dict['grad_weight'], data_dict['serr'])
    ):
        # if i >= 5:
        #     break
        assert data_this.shape == (81,)
        assert weight_this.shape == (81, 32) == grad_this.shape
        assert np.isscalar(serr_this)
        print(i)
        # ok. let's compute cost.
        # model.linear_module.weight.data[...] = Tensor(weight_this)

        weight_this_actual = model.linear_module.weight.data.cpu().numpy()
        assert weight_this_actual.shape == (81, 32)
        print(norm(weight_this_actual.ravel() - weight_this.ravel()) / (norm(weight_this.ravel()) + 1e-6))

        optimizer.zero_grad()
        input_this = Tensor(data_this[np.newaxis])
        if gpu:
            input_this = input_this.cuda()
        cost_this, _ = model.forward(Variable(input_this))
        cost_this.backward()
        cost_this = cost_this.item()
        assert np.isscalar(cost_this)
        print(cost_this, serr_this)

        # check grad
        grad_this_actual = model.linear_module.weight.grad.data.cpu().numpy()
        assert grad_this_actual.shape == (81, 32)
        print(norm(grad_this_actual.ravel() - grad_this.ravel()) / (norm(grad_this.ravel()) + 1e-6))

        optimizer.step()


if __name__ == '__main__':
    gpu = len(argv) > 1
    if gpu:
        print('GPU support!')
    demo(gpu=gpu)

    # sample output.
    # if `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, sometimes more "accurate" results can be obtained.
    # (well this depends on how blas for lua torch is implemented).
    #
    # 0
    # 2.61611268749e-08
    # 43.1805 43.1805
    # 2.64659647327e-07
    # 1
    # 3.26278911902e-08
    # 20.2959 20.2959
    # 1.71575460968e-07
    # 2
    # 3.47194842761e-08
    # 36.0171 36.0171
    # 0.00117025386274
    # 3
    # 1.40639701086e-06
    # 3.54644 3.54645
    # 0.00821431432149
    # 4
    # 1.4283722749e-06
    # 5.21084 5.21084
    # 9.53782483197e-07
    # 5
    # 1.42836149266e-06
    # 3.53002 3.53002
    # 0.0
    # 6
    # 1.42836149266e-06
    # 11.3661 11.3661
    # 0.00478018666745
    # 7
    # 1.58112279994e-06
    # 40.906 40.9057
    # 0.0102877893582
    # 8
    # 1.24102502393e-05
    # 33.7172 33.7172
    # 0.0011931663148
    # 9
    # 1.24360109692e-05
    # 24.2264 24.2264
    # 0.00237716430219

    # GPU version
    #
    # GPU support!
    # 0
    # 2.61611268749e-08
    # 43.1805 43.1805
    # 1.60780196461e-07
    # 1
    # 3.26428155256e-08
    # 20.2959 20.2959
    # 2.1432005255e-07
    # 2
    # 3.47192396336e-08
    # 36.0171 36.0171
    # 1.61258931958e-07
    # 3
    # 3.80074572723e-08
    # 3.54645 3.54645
    # 5.80048322074e-07
    # 4
    # 3.84934352058e-08
    # 5.21084 5.21084
    # 3.09069096599e-07
    # 5
    # 3.85534676656e-08
    # 3.53002 3.53002
    # 0.0
    # 6
    # 3.85534676656e-08
    # 11.3661 11.3661
    # 1.0749870643e-06
    # 7
    # 4.0654224229e-08
    # 40.9057 40.9057
    # 2.22819292923e-07
    # 8
    # 4.47181734167e-08
    # 33.7172 33.7172
    # 2.38714051451e-07
    # 9
    # 4.80621791835e-08
    # 24.2264 24.2264
    # 0.000408876993549

    demo(data_file_prefix='demo_fista_debug_lam5', lam=0.5, gpu=gpu)

    # sample output
    # 0
    # 2.61611268749e-08
    # 37.35 37.35
    # 1.79564954089e-07
    # 1
    # 3.4675322279e-08
    # 18.2635 18.2635
    # 1.8769624477e-07
    # 2
    # 4.02489925743e-08
    # 32.3235 32.3235
    # 0.0010648294056
    # 3
    # 2.09694794192e-06
    # 3.25817 3.25769
    # 0.0173171483983
    # 4
    # 2.99907216578e-06
    # 4.98615 4.98611
    # 0.00578583219375
    # 5
    # 3.08702354721e-06
    # 3.46744 3.46746
    # 0.00848901201271
    # 6
    # 3.08635134954e-06
    # 10.3149 10.3149
    # 0.0047616865181
    # 7
    # 3.72570341912e-06
    # 36.8153 36.8153
    # 0.00493065128216
    # 8
    # 1.06865952261e-05
    # 30.59 30.5901
    # 0.00179251713591
    # 9
    # 1.09624497227e-05
    # 20.8953 20.9011
    # 0.0285488980698
    #

    # GPU version.
    # 0
    # 2.61611268749e-08
    # 37.35 37.35
    # 2.04642019804e-07
    # 1
    # 3.47106788726e-08
    # 18.2635 18.2635
    # 2.55263152676e-07
    # 2
    # 4.03776223343e-08
    # 32.3235 32.3235
    # 1.69859962006e-07
    # 3
    # 4.53440785534e-08
    # 3.25769 3.25769
    # 2.1986776932e-07
    # 4
    # 4.65728424339e-08
    # 4.98611 4.98611
    # 2.67140204568e-07
    # 5
    # 4.95220334287e-08
    # 3.46746 3.46746
    # 1.94738366219e-07
    # 6
    # 5.06349208783e-08
    # 10.3149 10.3149
    # 2.21694368105e-07
    # 7
    # 5.33795376612e-08
    # 36.8153 36.8153
    # 2.01568691096e-07
    # 8
    # 5.79902454788e-08
    # 30.5901 30.5901
    # 1.36878011556e-07
    # 9
    # 6.09854473707e-08
    # 20.9011 20.9011
    # 1.92646698354e-07
