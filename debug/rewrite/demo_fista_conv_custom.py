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
        cost_this = cost_this.item()
        assert np.isscalar(cost_this)
        print(cost_this, serr_this)

        # check grad
        grad_this_actual = model.linear_module.weight.grad.data.cpu().numpy()
        assert grad_this_actual.shape == (4, 1, 9, 9)  # （4,1,9,9) and (4,9,9) are compatible for ravel().
        print('grad norm', norm(grad_this_actual.ravel()))
        print('code norm', norm(_.data.cpu().numpy()))
        print(norm(grad_this_actual.ravel() - grad_this.ravel()) / (norm(grad_this.ravel()) + 1e-6))

        optimizer.step()


if __name__ == '__main__':
    gpu = len(argv) > 1
    if gpu:
        print('GPU support!')
    print('lam=0.1')
    demo(gpu=gpu)
    # sample output on Mac (CPU, openblas + pytorch pip, no flags like OMP_NUM_THREADS=1 MKL_NUM_THREADS=1,
    # I wonder if MKL won't work fully on Mac)
    #
    # lam=0.1
    # 0
    # 2.64262845059e-08
    # 13.856 13.856
    # grad norm 15.7953
    # code norm 7.13257
    # 1.21668214924e-06
    # 1
    # 4.03109771703e-08
    # 3.87481 3.87481
    # grad norm 3.55712
    # code norm 2.18972
    # 5.64018319604e-07
    # 2
    # 4.69838383577e-08
    # 2.66957 2.66957
    # grad norm 1.80302
    # code norm 1.50991
    # 3.14920496577e-07
    # 3
    # 5.08647540775e-08
    # 6.69116 6.69115
    # grad norm 5.18855
    # code norm 3.49788
    # 1.03364193167e-06
    # 4
    # 5.72259984115e-08
    # 2.12411 2.12411
    # grad norm 1.76054
    # code norm 1.19931
    # 4.20385626156e-07
    # 5
    # 6.50616839409e-08
    # 8.3021 8.30209
    # grad norm 6.57363
    # code norm 4.54102
    # 1.04485922017e-06
    # 6
    # 7.12520580747e-08
    # 7.59268 7.59268
    # grad norm 8.90799
    # code norm 3.97625
    # 5.67808073407e-07
    # 7
    # 7.49315112639e-08
    # 7.99279 7.99279
    # grad norm 11.9581
    # code norm 5.03525
    # 9.11180999756e-07
    # 8
    # 7.92741758762e-08
    # 14.9272 14.9272
    # grad norm 25.2324
    # code norm 8.48327
    # 6.99627294694e-07
    # 9
    # 8.76489586223e-08
    # 1.22348 1.22348
    # grad norm 0.594887
    # code norm 0.559005
    # 2.85463580672e-07
    #
    #
    # GPU support!
    # lam=0.1
    # 0
    # 2.64262845059e-08
    # 13.856 13.856
    # grad norm 15.7953
    # code norm 7.13257
    # 1.22678365263e-06
    # 1
    # 3.90343612002e-08
    # 3.87481 3.87481
    # grad norm 3.55712
    # code norm 2.18972
    # 5.41672901286e-07
    # 2
    # 4.66508798526e-08
    # 2.66957 2.66957
    # grad norm 1.80302
    # code norm 1.50991
    # 3.12006408045e-07
    # 3
    # 5.13072408973e-08
    # 6.69116 6.69115
    # grad norm 5.18855
    # code norm 3.49788
    # 1.1433196648e-06
    # 4
    # 5.84287321559e-08
    # 2.12411 2.12411
    # grad norm 1.76054
    # code norm 1.19931
    # 3.64043498887e-07
    # 5
    # 6.5361590606e-08
    # 8.30209 8.30209
    # grad norm 6.57363
    # code norm 4.54102
    # 1.0326231191e-06
    # 6
    # 7.08833939516e-08
    # 7.59268 7.59268
    # grad norm 8.90799
    # code norm 3.97625
    # 5.89974799538e-07
    # 7
    # 7.10896704469e-08
    # 7.99279 7.99279
    # grad norm 11.9581
    # code norm 5.03525
    # 9.76518242993e-07
    # 8
    # 7.53842600427e-08
    # 14.9272 14.9272
    # grad norm 25.2324
    # code norm 8.48327
    # 7.82432590764e-07
    # 9
    # 8.39199335503e-08
    # 1.22348 1.22348
    # grad norm 0.594887
    # code norm 0.559005
    # 2.50092765138e-07

    # lam=1. not very accurate (for CPU).
    print('lam=1')
    # when code norm is zero, grad is zero. check matrix cookbook.
    # because gradient w.r.t. weight involves multiplication of code.
    # so a zero code vanishes everything.
    demo(gpu=gpu, lam=1.0, data_file_prefix='demo_fista_conv_debug_lam1')
    #
    # CPU
    # lam=1
    # 0
    # 2.64262845059e-08
    # 45.1018 45.1018
    # grad norm 10.4752
    # code norm 1.85997
    # 4.40675410137e-07
    # 1
    # 3.85276956445e-08
    # 7.68786 7.68786
    # grad norm 0.0
    # code norm 0.0
    # 0.0
    # 2
    # 3.85276956445e-08
    # 4.28166 4.28166
    # grad norm 0.0
    # code norm 0.0
    # 0.0
    # 3
    # 3.85276956445e-08
    # 15.0336 15.0336
    # grad norm 0.208314
    # code norm 0.0560567
    # 0.0271547562773
    # 4
    # 7.23338546732e-06
    # 3.27799 3.27801
    # grad norm 0.0
    # code norm 0.0
    # 0.0
    # 5
    # 7.23338546732e-06
    # 24.1734 24.1733
    # grad norm 3.17928
    # code norm 0.693929
    # 0.0113521514113
    # 6
    # 4.53615136954e-05
    # 21.0469 21.0469
    # grad norm 0.786346
    # code norm 0.165071
    # 0.0238666274195
    # 7
    # 4.99246804839e-05
    # 24.4177 24.4177
    # grad norm 4.1853
    # code norm 0.58977
    # 0.00219019013246
    # 8
    # 5.10640988435e-05
    # 52.4748 52.4747
    # grad norm 14.4638
    # code norm 1.45672
    # 0.00151356795254
    # 9
    # 5.84796080809e-05
    # 1.44808 1.44806
    # grad norm 0.0
    # code norm 0.0
    # 0.0
    #
    #
    #
    # GPU
    # lam=1
    # 0
    # 2.64262845059e-08
    # 45.1018 45.1018
    # grad norm 10.4752
    # code norm 1.85997
    # 5.97506834182e-07
    # 1
    # 3.66186825244e-08
    # 7.68786 7.68786
    # grad norm 0.0
    # code norm 0.0
    # 0.0
    # 2
    # 3.66186825244e-08
    # 4.28166 4.28166
    # grad norm 0.0
    # code norm 0.0
    # 0.0
    # 3
    # 3.66186825244e-08
    # 15.0336 15.0336
    # grad norm 0.214128
    # code norm 0.0576278
    # 3.13885109871e-07
    # 4
    # 3.72725047346e-08
    # 3.27801 3.27801
    # grad norm 0.0
    # code norm 0.0
    # 0.0
    # 5
    # 3.72725047346e-08
    # 24.1733 24.1733
    # grad norm 3.17577
    # code norm 0.692818
    # 4.12905303449e-07
    # 6
    # 4.65649654887e-08
    # 21.0469 21.0469
    # grad norm 0.770281
    # code norm 0.162708
    # 1.73217727121e-06
    # 7
    # 4.8931867035e-08
    # 24.4177 24.4177
    # grad norm 4.18129
    # code norm 0.589581
    # 4.29616703212e-07
    # 8
    # 5.12816851646e-08
    # 52.4747 52.4747
    # grad norm 14.4626
    # code norm 1.45578
    # 3.85032186578e-07
    # 9
    # 5.66612538937e-08
    # 1.44806 1.44806
    # grad norm 0.0
    # code norm 0.0
    # 0.0
