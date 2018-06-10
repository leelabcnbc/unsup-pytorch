"""this file tries to replicate https://github.com/koraykv/unsup/blob/master/demo/demo_fista.lua,
with its original experiment parameters.
the original script's data for first 10 iterations were collected using `/debug/reference/demo_fista_debug.lua`


run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python demo_fista_custom.py` to get full reproducibility.
https://discuss.pytorch.org/t/nondeterministic-behaviour-of-grad-running-on-cpu/7402/2
"""
import os
from sys import argv
import numpy as np
import spams
from numpy.linalg import norm
from unsup import dir_dictionary, psd  # somehow I need to import spams earlier than torch.
from torch import Tensor, optim
from torch.autograd import Variable


def demo(data_dir=dir_dictionary['debug_reference'],
         data_file_prefix='demo_psd_debug', lam=1.0, beta=1.0, gpu=False):
    # load all data.
    data_dict = {
        k: np.load(os.path.join(data_dir, f'{data_file_prefix}_{k}.npy')) for k in ('data', 'serr',
                                                                                    'weight', 'grad_weight',
                                                                                    'encoder_weight', 'encoder_bias')}
    # print(data_dict['weight'].shape)
    num_data = data_dict['data'].shape[0]
    assert num_data == data_dict['serr'].shape[0] == data_dict['weight'].shape[0] == data_dict['grad_weight'].shape[0]
    assert num_data == data_dict['encoder_weight'].shape[0] == data_dict['encoder_bias'].shape[0]

    model = psd.LinearPSD(81, 32, lam, beta)
    # intialize.
    if gpu:
        model.cuda()

    init_weight = Tensor(data_dict['weight'][0])
    init_encoder_weight = Tensor(data_dict['encoder_weight'][0])
    init_encoder_bias = Tensor(data_dict['encoder_bias'][0])
    assert model.decoder.linear_module.weight.size() == init_weight.size()
    assert model.encoder[0].weight.size() == init_encoder_weight.size()
    assert model.encoder[0].bias.size() == init_encoder_bias.size()
    if gpu:
        init_weight = init_weight.cuda()
        init_encoder_weight = init_encoder_weight.cuda()
        init_encoder_bias = init_encoder_bias.cuda()
    model.decoder.linear_module.weight.data[...] = init_weight
    model.encoder[0].weight.data[...] = init_encoder_weight
    model.encoder[0].bias.data[...] = init_encoder_bias
    # change factor from 32 to 81 or 1 will dramatically increase the difference between ref and actual.
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # let's have the for loop
    for i, (data_this, weight_this, grad_this, serr_this, encoder_weight_this, encoder_bias_this) in enumerate(
            zip(data_dict['data'], data_dict['weight'], data_dict['grad_weight'], data_dict['serr'],
                data_dict['encoder_weight'], data_dict['encoder_bias'])
    ):
        # if i >= 5:
        #     break
        assert data_this.shape == (81,)
        assert weight_this.shape == (81, 32) == grad_this.shape
        assert encoder_weight_this.shape == (32, 81)
        assert encoder_bias_this.shape == (32,)
        assert np.isscalar(serr_this)
        print(i)
        # ok. let's compute cost.

        # compare real updated and ref updated.
        weight_this_actual = model.decoder.linear_module.weight.data.cpu().numpy()
        assert weight_this_actual.shape == (81, 32)
        print('decoder weight',
              norm(weight_this_actual.ravel() - weight_this.ravel()) / (norm(weight_this.ravel()) + 1e-6))

        enc_weight_this_actual = model.encoder[0].weight.data.cpu().numpy()
        assert enc_weight_this_actual.shape == (32, 81)
        print('encoder weight',
              norm(enc_weight_this_actual.ravel() - encoder_weight_this.ravel()) / (
                      norm(encoder_weight_this.ravel()) + 1e-6))

        enc_bias_this_actual = model.encoder[0].bias.data.cpu().numpy()
        assert enc_bias_this_actual.shape == (32,)
        print('decoder bias',
              norm(enc_bias_this_actual.ravel() - encoder_bias_this.ravel()) / (norm(encoder_bias_this.ravel()) + 1e-6))

        # update, using standard.
        print('use new weight')
        model.decoder.linear_module.weight.data[...] = Tensor(weight_this) if not gpu else Tensor(weight_this).cuda()
        model.encoder[0].weight.data[...] = Tensor(encoder_weight_this) if not gpu else Tensor(
            encoder_weight_this).cuda()
        model.encoder[0].bias.data[...] = Tensor(encoder_bias_this) if not gpu else Tensor(encoder_bias_this).cuda()

        # again

        # compare real updated and ref updated.
        weight_this_actual = model.decoder.linear_module.weight.data.cpu().numpy()
        assert weight_this_actual.shape == (81, 32)
        print('decoder weight',
              norm(weight_this_actual.ravel() - weight_this.ravel()) / (norm(weight_this.ravel()) + 1e-6))

        enc_weight_this_actual = model.encoder[0].weight.data.cpu().numpy()
        assert enc_weight_this_actual.shape == (32, 81)
        print('encoder weight',
              norm(enc_weight_this_actual.ravel() - encoder_weight_this.ravel()) / (
                      norm(encoder_weight_this.ravel()) + 1e-6))

        enc_bias_this_actual = model.encoder[0].bias.data.cpu().numpy()
        assert enc_bias_this_actual.shape == (32,)
        print('decoder bias',
              norm(enc_bias_this_actual.ravel() - encoder_bias_this.ravel()) / (norm(encoder_bias_this.ravel()) + 1e-6))

        optimizer.zero_grad()
        input_this = Tensor(data_this[np.newaxis])
        if gpu:
            input_this = input_this.cuda()
        cost_this = model.forward(Variable(input_this))
        cost_this.backward()
        cost_this = cost_this.data.cpu().numpy()
        assert cost_this.shape == (1,)
        cost_this = cost_this[0]
        print(cost_this, serr_this)

        # check grad
        grad_this_actual = model.decoder.linear_module.weight.grad.data.cpu().numpy()
        assert grad_this_actual.shape == (81, 32)
        print(norm(grad_this_actual.ravel() - grad_this.ravel()) / (norm(grad_this.ravel()) + 1e-6))

        optimizer.step()


if __name__ == '__main__':
    gpu = len(argv) > 1
    if gpu:
        print('GPU support!')
    print('beta=1')
    demo(gpu=gpu)
    print('beta=5')
    # demo(data_file_prefix='demo_psd_debug_lam5', lam=0.5, gpu=gpu)
    demo(data_file_prefix='demo_psd_debug_beta5', beta=5, gpu=gpu)
    print('beta=5, with beta=1 data')
    # I believe weight of encoder won't change.
    demo(beta=5, gpu=gpu)
