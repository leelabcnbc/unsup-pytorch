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
         data_file_prefix='demo_psd_conv_debug', lam=1.0, beta=1.0, gpu=False):
    # load all data.
    data_dict = {
        k: np.load(os.path.join(data_dir, f'{data_file_prefix}_{k}.npy')) for k in ('data', 'serr',
                                                                                    'weight', 'grad_weight',
                                                                                    'encoder_weight', 'encoder_bias',
                                                                                    'encoder_scale')}
    # print(data_dict['weight'].shape)
    num_data = data_dict['data'].shape[0]
    assert num_data == data_dict['serr'].shape[0] == data_dict['weight'].shape[0] == data_dict['grad_weight'].shape[0]
    assert num_data == data_dict['encoder_weight'].shape[0] == data_dict['encoder_bias'].shape[0] == \
           data_dict['encoder_scale'].shape[0]

    model = psd.ConvPSD(4, 9, lam, beta, 25, True)
    # intialize.
    if gpu:
        model.cuda()

    init_weight = Tensor(data_dict['weight'][0][:, np.newaxis])
    init_encoder_weight = Tensor(data_dict['encoder_weight'][0][:, np.newaxis])
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
    optimizer = optim.SGD(model.parameters(), lr=0.002)

    # let's have the for loop
    for i, (data_this, weight_this, grad_this, serr_this, encoder_weight_this, encoder_bias_this,
            encoder_scale_this) in enumerate(
        zip(data_dict['data'], data_dict['weight'], data_dict['grad_weight'], data_dict['serr'],
            data_dict['encoder_weight'], data_dict['encoder_bias'], data_dict['encoder_scale'])
    ):
        # if i >= 5:
        #     break
        assert data_this.shape == (1, 25, 25)
        assert weight_this.shape == (4, 9, 9) == grad_this.shape
        assert encoder_weight_this.shape == (4, 9, 9)
        assert encoder_bias_this.shape == (4,)
        assert encoder_scale_this.shape == (4,)
        assert np.isscalar(serr_this)
        print(i)
        # ok. let's compute cost.

        # compare real updated and ref updated.
        weight_this_actual = model.decoder.linear_module.weight.data.cpu().numpy()
        assert weight_this_actual.shape == (4, 1, 9, 9)
        print('decoder weight',
              norm(weight_this_actual.ravel() - weight_this.ravel()) / (norm(weight_this.ravel()) + 1e-6))

        enc_weight_this_actual = model.encoder[0].weight.data.cpu().numpy()
        assert enc_weight_this_actual.shape == (4, 1, 9, 9)
        print('encoder weight',
              norm(enc_weight_this_actual.ravel() - encoder_weight_this.ravel()) / (
                      norm(encoder_weight_this.ravel()) + 1e-6))

        enc_bias_this_actual = model.encoder[0].bias.data.cpu().numpy()
        assert enc_bias_this_actual.shape == (4,)
        print('encoder bias',
              norm(enc_bias_this_actual.ravel() - encoder_bias_this.ravel()) / (norm(encoder_bias_this.ravel()) + 1e-6))

        enc_scale_this_actual = model.encoder[2].weight.data.cpu().numpy()
        assert enc_scale_this_actual.shape == (4, 1, 1)
        print('encoder scale',
              norm(enc_scale_this_actual.ravel() - encoder_scale_this.ravel()) / (
                          norm(encoder_scale_this.ravel()) + 1e-6))

        # update, using standard.
        print('use new weight')
        model.decoder.linear_module.weight.data[...] = Tensor(weight_this[:, np.newaxis]) if not gpu else Tensor(
            weight_this[:, np.newaxis]).cuda()
        model.encoder[0].weight.data[...] = Tensor(encoder_weight_this[:, np.newaxis]) if not gpu else Tensor(
            encoder_weight_this[:, np.newaxis]).cuda()
        model.encoder[0].bias.data[...] = Tensor(encoder_bias_this) if not gpu else Tensor(encoder_bias_this).cuda()
        model.encoder[2].weight.data[...] = Tensor(encoder_scale_this[:,
                                                  np.newaxis, np.newaxis]) if not gpu else Tensor(encoder_scale_this[:,
                                                                                                  np.newaxis,
                                                                                                  np.newaxis]).cuda()

        # again

        # compare real updated and ref updated.
        weight_this_actual = model.decoder.linear_module.weight.data.cpu().numpy()
        assert weight_this_actual.shape == (4, 1, 9, 9)
        print('decoder weight',
              norm(weight_this_actual.ravel() - weight_this.ravel()) / (norm(weight_this.ravel()) + 1e-6))

        enc_weight_this_actual = model.encoder[0].weight.data.cpu().numpy()
        assert enc_weight_this_actual.shape == (4, 1, 9, 9)
        print('encoder weight',
              norm(enc_weight_this_actual.ravel() - encoder_weight_this.ravel()) / (
                      norm(encoder_weight_this.ravel()) + 1e-6))

        enc_bias_this_actual = model.encoder[0].bias.data.cpu().numpy()
        assert enc_bias_this_actual.shape == (4,)
        print('encoder bias',
              norm(enc_bias_this_actual.ravel() - encoder_bias_this.ravel()) / (norm(encoder_bias_this.ravel()) + 1e-6))

        enc_scale_this_actual = model.encoder[2].weight.data.cpu().numpy()
        assert enc_scale_this_actual.shape == (4, 1, 1)
        print('encoder scale',
              norm(enc_scale_this_actual.ravel() - encoder_scale_this.ravel()) / (
                      norm(encoder_scale_this.ravel()) + 1e-6))


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
        assert grad_this_actual.shape == (4, 1, 9, 9)
        print(norm(grad_this_actual.ravel() - grad_this.ravel()) / (norm(grad_this.ravel()) + 1e-6))

        optimizer.step()

        # then normalize
        model.normalize()


if __name__ == '__main__':
    gpu = len(argv) > 1
    if gpu:
        print('GPU support!')
    print('beta=1')
    demo(gpu=gpu)
    print('beta=5')
    demo(data_file_prefix='demo_psd_conv_debug_beta5', beta=5, gpu=gpu)
    print('beta=5, with beta=1 data')
    # # I believe weight of encoder won't change.
    demo(beta=5, gpu=gpu)

    # demo CPU
    #
    # beta=1
    # 0
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # use new weight
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # 122.84 122.84
    # 0.0
    # 1
    # decoder weight 5.15356677392e-08
    # encoder weight 8.74862833293e-08
    # encoder bias 1.63676325187e-06
    # encoder scale 2.55852798335e-08
    # use new weight
    # decoder weight 2.70478036425e-08
    # encoder weight 2.48516189741e-08
    # encoder bias 2.00220855557e-08
    # encoder scale 2.55852798335e-08
    # 127.738 127.738
    # 0.00102346926688
    # 2
    # decoder weight 1.32277987335e-05
    # encoder weight 6.85130217704e-05
    # encoder bias 0.000417440272752
    # encoder scale 1.37849385621e-06
    # use new weight
    # decoder weight 2.53648272569e-08
    # encoder weight 2.34496259255e-08
    # encoder bias 3.93445373497e-08
    # encoder scale 1.78574355539e-08
    # 32.1791 32.1605
    # 0.0407681318853
    # 3
    # decoder weight 0.000105214692841
    # encoder weight 0.000704071148651
    # encoder bias 0.00963621753149
    # encoder scale 2.3407841854e-06
    # use new weight
    # decoder weight 2.61191425815e-08
    # encoder weight 2.41943359128e-08
    # encoder bias 3.8224762561e-08
    # encoder scale 2.00755453348e-08
    # 14.383 14.3833
    # 0.0
    # 4
    # decoder weight 2.611914275e-08
    # encoder weight 4.44419709378e-08
    # encoder bias 2.73718413996e-07
    # encoder scale 2.58099551646e-08
    # use new weight
    # decoder weight 2.611914275e-08
    # encoder weight 2.31157205782e-08
    # encoder bias 1.66480500018e-08
    # encoder scale 2.58099551646e-08
    # 100.266 100.267
    # 3.39817313815e-05
    # 5
    # decoder weight 4.67149122408e-07
    # encoder weight 3.33641520323e-06
    # encoder bias 0.000491320195662
    # encoder scale 6.37841680105e-08
    # use new weight
    # decoder weight 2.49293398415e-08
    # encoder weight 2.56818935807e-08
    # encoder bias 1.33578908505e-08
    # encoder scale 2.39746048592e-08
    # 6.76164 6.76164
    # 0.0
    # 6
    # decoder weight 2.49293387363e-08
    # encoder weight 3.43444209694e-08
    # encoder bias 1.22772508058e-07
    # encoder scale 2.72785460046e-08
    # use new weight
    # decoder weight 2.49293387363e-08
    # encoder weight 2.49699809922e-08
    # encoder bias 2.77271044643e-08
    # encoder scale 2.42133926617e-08
    # 9.0808 9.08079
    # 0.0
    # 7
    # decoder weight 2.49293386947e-08
    # encoder weight 3.94993475442e-08
    # encoder bias 2.34655206033e-07
    # encoder scale 2.78135103884e-08
    # use new weight
    # decoder weight 2.49293386947e-08
    # encoder weight 2.54997816265e-08
    # encoder bias 3.1592016327e-08
    # encoder scale 2.63484968479e-08
    # 7.42881 7.42882
    # 1.5785402494e-06
    # 8
    # decoder weight 3.17305873872e-08
    # encoder weight 3.57585703279e-08
    # encoder bias 4.46775875199e-07
    # encoder scale 4.39337390058e-08
    # use new weight
    # decoder weight 2.3904559186e-08
    # encoder weight 2.38996378788e-08
    # encoder bias 3.60340859671e-08
    # encoder scale 2.57684614927e-08
    # 19.7415 19.7412
    # 0.00362271598936
    # 9
    # decoder weight 2.34338656634e-06
    # encoder weight 1.53518337982e-05
    # encoder bias 0.000333665469435
    # encoder scale 4.31801413205e-08
    # use new weight
    # decoder weight 2.50405999304e-08
    # encoder weight 2.60995359871e-08
    # encoder bias 6.62416901454e-09
    # encoder scale 1.40322487341e-08
    # 3.42809 3.4281
    # 0.0
    # beta=5
    # 0
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # use new weight
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # 499.89 499.891
    # 0.0
    # 1
    # decoder weight 5.15356677392e-08
    # encoder weight 2.13540238348e-07
    # encoder bias 2.39650931664e-07
    # encoder scale 5.59995773128e-08
    # use new weight
    # decoder weight 2.70478036425e-08
    # encoder weight 2.48520202622e-08
    # encoder bias 2.09128518594e-08
    # encoder scale 3.25800429673e-08
    # 393.312 393.292
    # 0.00102346926688
    # 2
    # decoder weight 1.32277987335e-05
    # encoder weight 8.43124898281e-05
    # encoder bias 5.72957146477e-05
    # encoder scale 0.000104754683865
    # use new weight
    # decoder weight 2.53648272569e-08
    # encoder weight 2.57512918353e-08
    # encoder bias 2.40097296968e-08
    # encoder scale 1.96318343327e-08
    # 108.118 108.016
    # 0.0407681318853
    # 3
    # decoder weight 0.000105214692841
    # encoder weight 9.06301080582e-05
    # encoder bias 0.000164665862076
    # encoder scale 3.64130555488e-05
    # use new weight
    # decoder weight 2.61191425815e-08
    # encoder weight 2.39315152943e-08
    # encoder bias 2.44103309086e-08
    # encoder scale 2.87012340577e-08
    # 36.9 36.9003
    # 0.0
    # 4
    # decoder weight 2.611914275e-08
    # encoder weight 3.90214999762e-08
    # encoder bias 3.83222500487e-07
    # encoder scale 2.26959384779e-08
    # use new weight
    # decoder weight 2.611914275e-08
    # encoder weight 2.5376110104e-08
    # encoder bias 1.63170493479e-08
    # encoder scale 2.25631983331e-08
    # 144.424 144.424
    # 3.39817313815e-05
    # 5
    # decoder weight 4.67149122408e-07
    # encoder weight 1.05904986025e-06
    # encoder bias 5.17683099173e-06
    # encoder scale 2.53015856872e-06
    # use new weight
    # decoder weight 2.49293398415e-08
    # encoder weight 2.45537109654e-08
    # encoder bias 2.76181335391e-08
    # encoder scale 1.49118629009e-09
    # 35.705 35.705
    # 0.0
    # 6
    # decoder weight 2.49293387363e-08
    # encoder weight 3.87946298994e-08
    # encoder bias 2.92198536079e-08
    # encoder scale 2.52421841516e-07
    # use new weight
    # decoder weight 2.49293387363e-08
    # encoder weight 2.46210306793e-08
    # encoder bias 2.28718103563e-08
    # encoder scale 3.60907267001e-08
    # 6.39488 6.39487
    # 0.0
    # 7
    # decoder weight 2.49293386947e-08
    # encoder weight 3.46100518838e-08
    # encoder bias 2.96173941657e-08
    # encoder scale 4.59058973585e-08
    # use new weight
    # decoder weight 2.49293386947e-08
    # encoder weight 2.56266124778e-08
    # encoder bias 1.88597354965e-08
    # encoder scale 3.14013611747e-08
    # 6.77319 6.77319
    # 1.5785402494e-06
    # 8
    # decoder weight 3.17305873872e-08
    # encoder weight 3.59135822664e-08
    # encoder bias 1.25162509713e-08
    # encoder scale 1.35720048995e-07
    # use new weight
    # decoder weight 2.3904559186e-08
    # encoder weight 2.61635173712e-08
    # encoder bias 1.25162509713e-08
    # encoder scale 2.14418717333e-08
    # 17.3806 17.3789
    # 0.00362271598936
    # 9
    # decoder weight 2.34338656634e-06
    # encoder weight 9.73558982002e-08
    # encoder bias 4.28208383014e-07
    # encoder scale 0.000519871997432
    # use new weight
    # decoder weight 2.50405999304e-08
    # encoder weight 2.65750743791e-08
    # encoder bias 2.59634318063e-08
    # encoder scale 2.26378978887e-08
    # 2.80134 2.80135
    # 0.0
    # beta=5, with beta=1 data
    # 0
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # use new weight
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # 499.89 122.84
    # 0.0
    # 1
    # decoder weight 5.15356677392e-08
    # encoder weight 1.76996763547
    # encoder bias 14.0656439679
    # encoder scale 0.426047535087
    # use new weight
    # decoder weight 2.70478036425e-08
    # encoder weight 2.48516189741e-08
    # encoder bias 2.00220855557e-08
    # encoder scale 2.55852798335e-08
    # 451.511 127.738
    # 0.00102346926688
    # 2
    # decoder weight 1.32277987335e-05
    # encoder weight 2.47494597861
    # encoder bias 6.54807697199
    # encoder scale 0.420599083786
    # use new weight
    # decoder weight 2.53648272569e-08
    # encoder weight 2.34496259255e-08
    # encoder bias 3.93445373497e-08
    # encoder scale 1.78574355539e-08
    # 84.972 32.1605
    # 0.0407681318853
    # 3
    # decoder weight 0.000105214692841
    # encoder weight 0.590595584905
    # encoder bias 3.65048272946
    # encoder scale 0.0796971100767
    # use new weight
    # decoder weight 2.61191425815e-08
    # encoder weight 2.41943359128e-08
    # encoder bias 3.8224762561e-08
    # encoder scale 2.00755453348e-08
    # 40.8852 14.3833
    # 0.0
    # 4
    # decoder weight 2.611914275e-08
    # encoder weight 0.57473363713
    # encoder bias 5.80724084048
    # encoder scale 0.0419415078373
    # use new weight
    # decoder weight 2.611914275e-08
    # encoder weight 2.31157205782e-08
    # encoder bias 1.66480500018e-08
    # encoder scale 2.58099551646e-08
    # 282.868 100.267
    # 3.39817313815e-05
    # 5
    # decoder weight 4.67149122408e-07
    # encoder weight 1.81790279212
    # encoder bias 18.5080334756
    # encoder scale 0.273490004267
    # use new weight
    # decoder weight 2.49293398415e-08
    # encoder weight 2.56818935807e-08
    # encoder bias 1.33578908505e-08
    # encoder scale 2.39746048592e-08
    # 9.73147 6.76164
    # 0.0
    # 6
    # decoder weight 2.49293387363e-08
    # encoder weight 0.160899536785
    # encoder bias 6.09734576362
    # encoder scale 0.00632202589661
    # use new weight
    # decoder weight 2.49293387363e-08
    # encoder weight 2.49699809922e-08
    # encoder bias 2.77271044643e-08
    # encoder scale 2.42133926617e-08
    # 19.8718 9.08079
    # 0.0
    # 7
    # decoder weight 2.49293386947e-08
    # encoder weight 0.22165864471
    # encoder bias 7.52167196689
    # encoder scale 0.0196445533366
    # use new weight
    # decoder weight 2.49293386947e-08
    # encoder weight 2.54997816265e-08
    # encoder bias 3.1592016327e-08
    # encoder scale 2.63484968479e-08
    # 11.6459 7.42882
    # 1.5785402494e-06
    # 8
    # decoder weight 3.17305873872e-08
    # encoder weight 0.168971791921
    # encoder bias 4.47782318049
    # encoder scale 0.00780360101525
    # use new weight
    # decoder weight 2.3904559186e-08
    # encoder weight 2.38996378788e-08
    # encoder bias 3.60340859671e-08
    # encoder scale 2.57684614927e-08
    # 30.2094 19.7412
    # 0.00362271598936
    # 9
    # decoder weight 2.34338656634e-06
    # encoder weight 0.393815267723
    # encoder bias 1.0123824648
    # encoder scale 0.0192129596881
    # use new weight
    # decoder weight 2.50405999304e-08
    # encoder weight 2.60995359871e-08
    # encoder bias 6.62416901454e-09
    # encoder scale 1.40322487341e-08
    # 5.94193 3.4281
    # 0.0
    #


    #
    #
    # GPU support!
    # beta=1
    # 0
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # use new weight
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # 122.84 122.84
    # 0.0
    # 1
    # decoder weight 7.10910682616e-08
    # encoder weight 5.85807246764e-08
    # encoder bias 2.45057781044e-07
    # encoder scale 2.55852798335e-08
    # use new weight
    # decoder weight 2.70478036425e-08
    # encoder weight 2.48516189741e-08
    # encoder bias 2.00220855557e-08
    # encoder scale 2.55852798335e-08
    # 127.738 127.738
    # 3.90454900312e-07
    # 2
    # decoder weight 5.36393379638e-08
    # encoder weight 8.94406146475e-08
    # encoder bias 3.66802582856e-07
    # encoder scale 3.96852306704e-08
    # use new weight
    # decoder weight 2.53648272569e-08
    # encoder weight 2.34496259255e-08
    # encoder bias 3.93445373497e-08
    # encoder scale 1.78574355539e-08
    # 32.1605 32.1605
    # 4.35480367704e-07
    # 3
    # decoder weight 3.16669445743e-08
    # encoder weight 3.89978436842e-08
    # encoder bias 1.81795193757e-07
    # encoder scale 2.90333220617e-08
    # use new weight
    # decoder weight 2.61191425815e-08
    # encoder weight 2.41943359128e-08
    # encoder bias 3.8224762561e-08
    # encoder scale 2.00755453348e-08
    # 14.3833 14.3833
    # 0.0
    # 4
    # decoder weight 2.611914275e-08
    # encoder weight 3.70176840319e-08
    # encoder bias 2.97183678443e-07
    # encoder scale 2.58099551646e-08
    # use new weight
    # decoder weight 2.611914275e-08
    # encoder weight 2.31157205782e-08
    # encoder bias 1.66480500018e-08
    # encoder scale 2.58099551646e-08
    # 100.267 100.267
    # 4.05539197395e-07
    # 5
    # decoder weight 6.12818013792e-08
    # encoder weight 7.61714807202e-08
    # encoder bias 2.07766603955e-06
    # encoder scale 3.66238846297e-08
    # use new weight
    # decoder weight 2.49293398415e-08
    # encoder weight 2.56818935807e-08
    # encoder bias 1.33578908505e-08
    # encoder scale 2.39746048592e-08
    # 6.76164 6.76164
    # 0.0
    # 6
    # decoder weight 2.49293387363e-08
    # encoder weight 3.4801102345e-08
    # encoder bias 3.8450597346e-07
    # encoder scale 2.72785460046e-08
    # use new weight
    # decoder weight 2.49293387363e-08
    # encoder weight 2.49699809922e-08
    # encoder bias 2.77271044643e-08
    # encoder scale 2.42133926617e-08
    # 9.08079 9.08079
    # 0.0
    # 7
    # decoder weight 2.49293386947e-08
    # encoder weight 4.05638220151e-08
    # encoder bias 2.37124579311e-07
    # encoder scale 2.78135103884e-08
    # use new weight
    # decoder weight 2.49293386947e-08
    # encoder weight 2.54997816265e-08
    # encoder bias 3.1592016327e-08
    # encoder scale 2.63484968479e-08
    # 7.42882 7.42882
    # 8.90764233101e-07
    # 8
    # decoder weight 5.55045390183e-08
    # encoder weight 3.59082901225e-08
    # encoder bias 3.76268942361e-07
    # encoder scale 4.39337390058e-08
    # use new weight
    # decoder weight 2.3904559186e-08
    # encoder weight 2.38996378788e-08
    # encoder bias 3.60340859671e-08
    # encoder scale 2.57684614927e-08
    # 19.7412 19.7412
    # 8.91049453922e-07
    # 9
    # decoder weight 3.06176261835e-08
    # encoder weight 3.62204300969e-08
    # encoder bias 1.02328808638e-07
    # encoder scale 1.40322487341e-08
    # use new weight
    # decoder weight 2.50405999304e-08
    # encoder weight 2.60995359871e-08
    # encoder bias 6.62416901454e-09
    # encoder scale 1.40322487341e-08
    # 3.4281 3.4281
    # 0.0
    # beta=5
    # 0
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # use new weight
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # 499.891 499.891
    # 0.0
    # 1
    # decoder weight 7.10910682616e-08
    # encoder weight 1.25381160861e-07
    # encoder bias 1.1874339275e-07
    # encoder scale 1.14368259356e-07
    # use new weight
    # decoder weight 2.70478036425e-08
    # encoder weight 2.48520202622e-08
    # encoder bias 2.09128518594e-08
    # encoder scale 3.25800429673e-08
    # 393.292 393.292
    # 4.2931264982e-07
    # 2
    # decoder weight 5.46095262607e-08
    # encoder weight 8.84215849464e-08
    # encoder bias 2.57769681381e-07
    # encoder scale 2.17645685254e-07
    # use new weight
    # decoder weight 2.53648272569e-08
    # encoder weight 2.57512918353e-08
    # encoder bias 2.40097296968e-08
    # encoder scale 1.96318343327e-08
    # 108.016 108.016
    # 3.62332453568e-07
    # 3
    # decoder weight 3.13405045168e-08
    # encoder weight 3.90535526635e-08
    # encoder bias 5.59320278174e-08
    # encoder scale 2.92937225688e-08
    # use new weight
    # decoder weight 2.61191425815e-08
    # encoder weight 2.39315152943e-08
    # encoder bias 2.44103309086e-08
    # encoder scale 2.87012340577e-08
    # 36.9003 36.9003
    # 0.0
    # 4
    # decoder weight 2.611914275e-08
    # encoder weight 3.81620771468e-08
    # encoder bias 1.25112568591e-07
    # encoder scale 2.26959384779e-08
    # use new weight
    # decoder weight 2.611914275e-08
    # encoder weight 2.5376110104e-08
    # encoder bias 1.63170493479e-08
    # encoder scale 2.25631983331e-08
    # 144.424 144.424
    # 3.52943247975e-07
    # 5
    # decoder weight 6.16880932613e-08
    # encoder weight 5.20798024991e-08
    # encoder bias 1.26633098903e-07
    # encoder scale 1.00146927408e-08
    # use new weight
    # decoder weight 2.49293398415e-08
    # encoder weight 2.45537109654e-08
    # encoder bias 2.76181335391e-08
    # encoder scale 1.49118629009e-09
    # 35.705 35.705
    # 0.0
    # 6
    # decoder weight 2.49293387363e-08
    # encoder weight 3.52779943464e-08
    # encoder bias 4.46356749117e-08
    # encoder scale 8.96828423347e-07
    # use new weight
    # decoder weight 2.49293387363e-08
    # encoder weight 2.46210306793e-08
    # encoder bias 2.28718103563e-08
    # encoder scale 3.60907267001e-08
    # 6.39487 6.39487
    # 0.0
    # 7
    # decoder weight 2.49293386947e-08
    # encoder weight 3.46100518838e-08
    # encoder bias 2.96173941657e-08
    # encoder scale 4.59058973585e-08
    # use new weight
    # decoder weight 2.49293386947e-08
    # encoder weight 2.56266124778e-08
    # encoder bias 1.88597354965e-08
    # encoder scale 3.14013611747e-08
    # 6.77319 6.77319
    # 8.86487523627e-07
    # 8
    # decoder weight 5.55045390183e-08
    # encoder weight 3.59135822664e-08
    # encoder bias 1.25162509713e-08
    # encoder scale 8.23915753531e-08
    # use new weight
    # decoder weight 2.3904559186e-08
    # encoder weight 2.61635173712e-08
    # encoder bias 1.25162509713e-08
    # encoder scale 2.14418717333e-08
    # 17.3789 17.3789
    # 8.91049453922e-07
    # 9
    # decoder weight 3.06176261835e-08
    # encoder weight 3.56202720974e-08
    # encoder bias 3.76615023596e-08
    # encoder scale 1.55119792195e-07
    # use new weight
    # decoder weight 2.50405999304e-08
    # encoder weight 2.65750743791e-08
    # encoder bias 2.59634318063e-08
    # encoder scale 2.26378978887e-08
    # 2.80135 2.80135
    # 0.0
    # beta=5, with beta=1 data
    # 0
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # use new weight
    # decoder weight 2.49178445926e-08
    # encoder weight 2.42939320884e-08
    # encoder bias 1.38443189139e-08
    # encoder scale 0.0
    # 499.891 122.84
    # 0.0
    # 1
    # decoder weight 7.10910682616e-08
    # encoder weight 1.76996765036
    # encoder bias 14.0656462279
    # encoder scale 0.426047564828
    # use new weight
    # decoder weight 2.70478036425e-08
    # encoder weight 2.48516189741e-08
    # encoder bias 2.00220855557e-08
    # encoder scale 2.55852798335e-08
    # 451.509 127.738
    # 3.33366811956e-07
    # 2
    # decoder weight 5.39947301765e-08
    # encoder weight 2.47500595301
    # encoder bias 6.54846647123
    # encoder scale 0.420604783399
    # use new weight
    # decoder weight 2.53648272569e-08
    # encoder weight 2.34496259255e-08
    # encoder bias 3.93445373497e-08
    # encoder scale 1.78574355539e-08
    # 84.8801 32.1605
    # 2.72727443906e-07
    # 3
    # decoder weight 3.19683177933e-08
    # encoder weight 0.590181511206
    # encoder bias 3.66277639493
    # encoder scale 0.0797035678423
    # use new weight
    # decoder weight 2.61191425815e-08
    # encoder weight 2.41943359128e-08
    # encoder bias 3.8224762561e-08
    # encoder scale 2.00755453348e-08
    # 40.8855 14.3833
    # 0.0
    # 4
    # decoder weight 2.611914275e-08
    # encoder weight 0.574733666299
    # encoder bias 5.80724041968
    # encoder scale 0.0419415078373
    # use new weight
    # decoder weight 2.611914275e-08
    # encoder weight 2.31157205782e-08
    # encoder bias 1.66480500018e-08
    # encoder scale 2.58099551646e-08
    # 282.868 100.267
    # 3.1202039423e-07
    # 5
    # decoder weight 6.05994305511e-08
    # encoder weight 1.81790533462
    # encoder bias 18.5064227363
    # encoder scale 0.273490160184
    # use new weight
    # decoder weight 2.49293398415e-08
    # encoder weight 2.56818935807e-08
    # encoder bias 1.33578908505e-08
    # encoder scale 2.39746048592e-08
    # 9.73147 6.76164
    # 0.0
    # 6
    # decoder weight 2.49293387363e-08
    # encoder weight 0.160899546142
    # encoder bias 6.09735164414
    # encoder scale 0.00632202589661
    # use new weight
    # decoder weight 2.49293387363e-08
    # encoder weight 2.49699809922e-08
    # encoder bias 2.77271044643e-08
    # encoder scale 2.42133926617e-08
    # 19.8718 9.08079
    # 0.0
    # 7
    # decoder weight 2.49293386947e-08
    # encoder weight 0.22165865438
    # encoder bias 7.52167076312
    # encoder scale 0.0196445533366
    # use new weight
    # decoder weight 2.49293386947e-08
    # encoder weight 2.54997816265e-08
    # encoder bias 3.1592016327e-08
    # encoder scale 2.63484968479e-08
    # 11.6459 7.42882
    # 8.89527968183e-07
    # 8
    # decoder weight 5.55045390183e-08
    # encoder weight 0.168971798131
    # encoder bias 4.47782215666
    # encoder scale 0.00780360101525
    # use new weight
    # decoder weight 2.3904559186e-08
    # encoder weight 2.38996378788e-08
    # encoder bias 3.60340859671e-08
    # encoder scale 2.57684614927e-08
    # 30.2078 19.7412
    # 8.91049453922e-07
    # 9
    # decoder weight 3.06176261835e-08
    # encoder weight 0.393811471479
    # encoder bias 1.01183042483
    # encoder scale 0.0192130717169
    # use new weight
    # decoder weight 2.50405999304e-08
    # encoder weight 2.60995359871e-08
    # encoder bias 6.62416901454e-09
    # encoder scale 1.40322487341e-08
    # 5.94193 3.4281
    # 0.0
