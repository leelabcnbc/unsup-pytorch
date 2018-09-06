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
        cost_this = cost_this.item()
        assert np.isscalar(cost_this)
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
    # I believe weight of decoder (sparse coding, not fast approximation)
    # won't change.
    demo(beta=5, gpu=gpu)

    # sample output
    # for some itearations, errors are bigger;
    # I assume all these are due to the peculiarity of data.

    # CPU data, run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
    # beta=1
    # 0
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # use new weight
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # 15.7765 15.7765
    # 3.51991428478e-07
    # 1
    # decoder weight 3.05550888676e-08
    # encoder weight 3.45585399895e-08
    # decoder bias 3.69713637601e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.42561773013e-08
    # decoder bias 2.56706309991e-08
    # 5.02094 5.02094
    # 0.0
    # 2
    # decoder weight 2.64054342698e-08
    # encoder weight 3.51562242404e-08
    # decoder bias 3.30738239608e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.41880621762e-08
    # decoder bias 2.50201084075e-08
    # 6.9851 6.9851
    # 0.0
    # 3
    # decoder weight 2.64054342698e-08
    # encoder weight 3.42153482987e-08
    # decoder bias 3.59864838103e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.43964141223e-08
    # decoder bias 2.38035478959e-08
    # 9.85412 9.85425
    # 0.00150685540909
    # 4
    # decoder weight 6.41925810953e-06
    # encoder weight 1.03160809836e-05
    # decoder bias 3.63337834651e-05
    # use new weight
    # decoder weight 2.64296556958e-08
    # encoder weight 2.3999104006e-08
    # decoder bias 2.49059144706e-08
    # 14.7351 14.7415
    # 0.0111931125241
    # 5
    # decoder weight 8.98474941752e-05
    # encoder weight 0.00015296121627
    # decoder bias 0.000442009917433
    # use new weight
    # decoder weight 2.61451235151e-08
    # encoder weight 2.3984876869e-08
    # decoder bias 2.81146839595e-08
    # 21.0313 21.0313
    # 3.66064288772e-07
    # 6
    # decoder weight 2.99284777045e-08
    # encoder weight 3.64939500936e-08
    # decoder bias 4.49200197936e-08
    # use new weight
    # decoder weight 2.62446303486e-08
    # encoder weight 2.47224072835e-08
    # decoder bias 3.05768801369e-08
    # 3.89914 3.89925
    # 0.00920963878309
    # 7
    # decoder weight 2.75628492488e-06
    # encoder weight 4.40145740077e-06
    # decoder bias 2.57033216763e-05
    # use new weight
    # decoder weight 2.63295467238e-08
    # encoder weight 2.44789707071e-08
    # decoder bias 2.20199791863e-08
    # 11.37 11.37
    # 7.24353401903e-07
    # 8
    # decoder weight 3.0410108148e-08
    # encoder weight 3.53229213568e-08
    # decoder bias 3.84252293026e-08
    # use new weight
    # decoder weight 2.66324169701e-08
    # encoder weight 2.43670111255e-08
    # decoder bias 2.67602953118e-08
    # 48.0082 48.0068
    # 0.00236801215333
    # 9
    # decoder weight 0.000111095815046
    # encoder weight 0.000196703458295
    # decoder bias 0.000319650736447
    # use new weight
    # decoder weight 2.72904499331e-08
    # encoder weight 2.43044516448e-08
    # decoder bias 2.17212788937e-08
    # 24.9716 25.4231
    # 0.80121450345
    # beta=5
    # 0
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # use new weight
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # 26.0068 26.0068
    # 3.51991428478e-07
    # 1
    # decoder weight 3.05550888676e-08
    # encoder weight 5.08577638866e-08
    # decoder bias 9.74929066619e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.37787987644e-08
    # decoder bias 2.06375459689e-08
    # 7.70346 7.70346
    # 0.0
    # 2
    # decoder weight 2.64054342698e-08
    # encoder weight 3.45621147084e-08
    # decoder bias 2.73474423038e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.45182542538e-08
    # decoder bias 2.3513893097e-08
    # 10.1368 10.1368
    # 0.0
    # 3
    # decoder weight 2.64054342698e-08
    # encoder weight 3.46075372919e-08
    # decoder bias 4.59126828564e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.40698886139e-08
    # decoder bias 3.58888352502e-08
    # 16.4689 16.4695
    # 0.00150685540909
    # 4
    # decoder weight 6.41925810953e-06
    # encoder weight 5.18862407183e-05
    # decoder bias 0.000210716088501
    # use new weight
    # decoder weight 2.64296556958e-08
    # encoder weight 2.49183125909e-08
    # decoder bias 2.78298352676e-08
    # 20.5906 20.6232
    # 0.0111931125241
    # 5
    # decoder weight 8.98474941752e-05
    # encoder weight 0.000768963451942
    # decoder bias 0.00263727322086
    # use new weight
    # decoder weight 2.61451235151e-08
    # encoder weight 2.44909856486e-08
    # decoder bias 2.51555214474e-08
    # 37.8417 37.8417
    # 3.66064288772e-07
    # 6
    # decoder weight 2.99284777045e-08
    # encoder weight 6.72925350584e-08
    # decoder bias 1.41760561998e-07
    # use new weight
    # decoder weight 2.62446303486e-08
    # encoder weight 2.46744302959e-08
    # decoder bias 2.66764390832e-08
    # 5.9768 5.97745
    # 0.00920963878309
    # 7
    # decoder weight 2.75628492488e-06
    # encoder weight 2.19056695481e-05
    # decoder bias 0.000133040949478
    # use new weight
    # decoder weight 2.63295467238e-08
    # encoder weight 2.44901876272e-08
    # decoder bias 1.76738747494e-08
    # 18.268 18.268
    # 7.24353401903e-07
    # 8
    # decoder weight 3.0410108148e-08
    # encoder weight 4.64324525085e-08
    # decoder bias 1.00196050195e-07
    # use new weight
    # decoder weight 2.66324169701e-08
    # encoder weight 2.49899290248e-08
    # decoder bias 2.34876413294e-08
    # 91.6931 91.6743
    # 0.00236801215333
    # 9
    # decoder weight 0.000111095815046
    # encoder weight 0.000860527812538
    # decoder bias 0.000965642746255
    # use new weight
    # decoder weight 2.72904499331e-08
    # encoder weight 2.50383891963e-08
    # decoder bias 2.63595084454e-08
    # 97.1717 98.073
    # 0.80121450345
    # beta=5, with beta=1 data
    # 0
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # use new weight
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # 26.0068 15.7765
    # 3.51991428478e-07
    # 1
    # decoder weight 3.05550888676e-08
    # encoder weight 0.145144677621
    # decoder bias 0.380324640394
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.42561773013e-08
    # decoder bias 2.56706309991e-08
    # 7.7988 5.02094
    # 0.0
    # 2
    # decoder weight 2.64054342698e-08
    # encoder weight 0.0421666715077
    # decoder bias 0.202650046628
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.41880621762e-08
    # decoder bias 2.50201084075e-08
    # 10.1879 6.9851
    # 0.0
    # 3
    # decoder weight 2.64054342698e-08
    # encoder weight 0.0541968290184
    # decoder bias 0.221826723912
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.43964141223e-08
    # decoder bias 2.38035478959e-08
    # 15.1079 9.85425
    # 0.00150685540909
    # 4
    # decoder weight 6.41925810953e-06
    # encoder weight 0.0823655710488
    # decoder bias 0.290095972184
    # use new weight
    # decoder weight 2.64296556958e-08
    # encoder weight 2.3999104006e-08
    # decoder bias 2.49059144706e-08
    # 20.8361 14.7415
    # 0.0111931125241
    # 5
    # decoder weight 8.98474941752e-05
    # encoder weight 0.110821348712
    # decoder bias 0.320229267475
    # use new weight
    # decoder weight 2.61451235151e-08
    # encoder weight 2.3984876869e-08
    # decoder bias 2.81146839595e-08
    # 39.3739 21.0313
    # 3.66064288772e-07
    # 6
    # decoder weight 2.99284777045e-08
    # encoder weight 0.221210371682
    # decoder bias 0.568965603669
    # use new weight
    # decoder weight 2.62446303486e-08
    # encoder weight 2.47224072835e-08
    # decoder bias 3.05768801369e-08
    # 5.38292 3.89925
    # 0.00920963878309
    # 7
    # decoder weight 2.75628492488e-06
    # encoder weight 0.0280981058885
    # decoder bias 0.164080607695
    # use new weight
    # decoder weight 2.63295467238e-08
    # encoder weight 2.44789707071e-08
    # decoder bias 2.20199791863e-08
    # 18.2893 11.37
    # 7.24353401903e-07
    # 8
    # decoder weight 3.0410108148e-08
    # encoder weight 0.101342604479
    # decoder bias 0.35128816468
    # use new weight
    # decoder weight 2.66324169701e-08
    # encoder weight 2.43670111255e-08
    # decoder bias 2.67602953118e-08
    # 82.7223 48.0068
    # 0.00236801215333
    # 9
    # decoder weight 0.000111095815046
    # encoder weight 0.476137600278
    # decoder bias 0.773749764251
    # use new weight
    # decoder weight 2.72904499331e-08
    # encoder weight 2.43044516448e-08
    # decoder bias 2.17212788937e-08
    # 41.2115 25.4231
    # 0.80121450345



    # GPU data.
    # GPU support!
    # beta=1
    # 0
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # use new weight
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # 15.7765 15.7765
    # 1.81943369983e-07
    # 1
    # decoder weight 3.01394465662e-08
    # encoder weight 3.37801771014e-08
    # decoder bias 3.36810479239e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.42561773013e-08
    # decoder bias 2.56706309991e-08
    # 5.02094 5.02094
    # 0.0
    # 2
    # decoder weight 2.64054342698e-08
    # encoder weight 3.51411991081e-08
    # decoder bias 3.2556117756e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.41880621762e-08
    # decoder bias 2.50201084075e-08
    # 6.9851 6.9851
    # 0.0
    # 3
    # decoder weight 2.64054342698e-08
    # encoder weight 3.43315468088e-08
    # decoder bias 3.48684819747e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.43964141223e-08
    # decoder bias 2.38035478959e-08
    # 9.85412 9.85425
    # 0.00150694577948
    # 4
    # decoder weight 6.41989330816e-06
    # encoder weight 1.03157344146e-05
    # decoder bias 3.63349639093e-05
    # use new weight
    # decoder weight 2.64296556958e-08
    # encoder weight 2.3999104006e-08
    # decoder bias 2.49059144706e-08
    # 14.7351 14.7415
    # 0.0111931399853
    # 5
    # decoder weight 8.98473663793e-05
    # encoder weight 0.000152962543149
    # decoder bias 0.000442002329987
    # use new weight
    # decoder weight 2.61451235151e-08
    # encoder weight 2.3984876869e-08
    # decoder bias 2.81146839595e-08
    # 21.0313 21.0313
    # 2.25851527192e-07
    # 6
    # decoder weight 2.96732857843e-08
    # encoder weight 3.60929507211e-08
    # decoder bias 4.09022970463e-08
    # use new weight
    # decoder weight 2.62446303486e-08
    # encoder weight 2.47224072835e-08
    # decoder bias 3.05768801369e-08
    # 3.89914 3.89925
    # 0.00920927685535
    # 7
    # decoder weight 2.75615963256e-06
    # encoder weight 4.40095801423e-06
    # decoder bias 2.57017530109e-05
    # use new weight
    # decoder weight 2.63295467238e-08
    # encoder weight 2.44789707071e-08
    # decoder bias 2.20199791863e-08
    # 11.37 11.37
    # 4.22910324294e-07
    # 8
    # decoder weight 3.0026087854e-08
    # encoder weight 3.49737890154e-08
    # decoder bias 3.52058324751e-08
    # use new weight
    # decoder weight 2.66324169701e-08
    # encoder weight 2.43670111255e-08
    # decoder bias 2.67602953118e-08
    # 48.0068 48.0068
    # 2.28855024558e-07
    # 9
    # decoder weight 3.350114275e-08
    # encoder weight 4.05693381425e-08
    # decoder bias 4.7277661584e-08
    # use new weight
    # decoder weight 2.72904499331e-08
    # encoder weight 2.43044516448e-08
    # decoder bias 2.17212788937e-08
    # 25.4231 25.4231
    # 3.64612508628e-07
    # beta=5
    # 0
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # use new weight
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # 26.0068 26.0068
    # 1.81943369983e-07
    # 1
    # decoder weight 3.01394465662e-08
    # encoder weight 4.23887833658e-08
    # decoder bias 6.22131619686e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.37787987644e-08
    # decoder bias 2.06375459689e-08
    # 7.70346 7.70346
    # 0.0
    # 2
    # decoder weight 2.64054342698e-08
    # encoder weight 3.46077381229e-08
    # decoder bias 3.18851120083e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.45182542538e-08
    # decoder bias 2.3513893097e-08
    # 10.1368 10.1368
    # 0.0
    # 3
    # decoder weight 2.64054342698e-08
    # encoder weight 3.52661296873e-08
    # decoder bias 4.79665820691e-08
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.40698886139e-08
    # decoder bias 3.58888352502e-08
    # 16.4689 16.4695
    # 0.00150694577948
    # 4
    # decoder weight 6.41989330816e-06
    # encoder weight 5.18832742363e-05
    # decoder bias 0.000210716090072
    # use new weight
    # decoder weight 2.64296556958e-08
    # encoder weight 2.49183125909e-08
    # decoder bias 2.78298352676e-08
    # 20.5906 20.6232
    # 0.0111931399853
    # 5
    # decoder weight 8.98473663793e-05
    # encoder weight 0.000768963373637
    # decoder bias 0.00263727023908
    # use new weight
    # decoder weight 2.61451235151e-08
    # encoder weight 2.44909856486e-08
    # decoder bias 2.51555214474e-08
    # 37.8417 37.8417
    # 2.25851527192e-07
    # 6
    # decoder weight 2.96732857843e-08
    # encoder weight 4.72402650806e-08
    # decoder bias 8.19395069492e-08
    # use new weight
    # decoder weight 2.62446303486e-08
    # encoder weight 2.46744302959e-08
    # decoder bias 2.66764390832e-08
    # 5.9768 5.97745
    # 0.00920927685535
    # 7
    # decoder weight 2.75615963256e-06
    # encoder weight 2.19039766414e-05
    # decoder bias 0.000133028774147
    # use new weight
    # decoder weight 2.63295467238e-08
    # encoder weight 2.44901876272e-08
    # decoder bias 1.76738747494e-08
    # 18.268 18.268
    # 4.22910324294e-07
    # 8
    # decoder weight 3.0026087854e-08
    # encoder weight 4.06998195747e-08
    # decoder bias 7.72648821603e-08
    # use new weight
    # decoder weight 2.66324169701e-08
    # encoder weight 2.49899290248e-08
    # decoder bias 2.34876413294e-08
    # 91.6743 91.6743
    # 2.28855024558e-07
    # 9
    # decoder weight 3.350114275e-08
    # encoder weight 1.00633051269e-07
    # decoder bias 1.03907135451e-07
    # use new weight
    # decoder weight 2.72904499331e-08
    # encoder weight 2.50383891963e-08
    # decoder bias 2.63595084454e-08
    # 98.073 98.073
    # 3.64612508628e-07
    # beta=5, with beta=1 data
    # 0
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # use new weight
    # decoder weight 2.65011176503e-08
    # encoder weight 2.43935529108e-08
    # decoder bias 2.4134509344e-08
    # 26.0068 15.7765
    # 1.81943369983e-07
    # 1
    # decoder weight 3.01394465662e-08
    # encoder weight 0.14514467603
    # decoder bias 0.380324633073
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.42561773013e-08
    # decoder bias 2.56706309991e-08
    # 7.79879 5.02094
    # 0.0
    # 2
    # decoder weight 2.64054342698e-08
    # encoder weight 0.0421666712623
    # decoder bias 0.202650047516
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.41880621762e-08
    # decoder bias 2.50201084075e-08
    # 10.1879 6.9851
    # 0.0
    # 3
    # decoder weight 2.64054342698e-08
    # encoder weight 0.0541968289919
    # decoder bias 0.221826728959
    # use new weight
    # decoder weight 2.64054342698e-08
    # encoder weight 2.43964141223e-08
    # decoder bias 2.38035478959e-08
    # 15.1079 9.85425
    # 0.00150694577948
    # 4
    # decoder weight 6.41989330816e-06
    # encoder weight 0.082365568717
    # decoder bias 0.290095961668
    # use new weight
    # decoder weight 2.64296556958e-08
    # encoder weight 2.3999104006e-08
    # decoder bias 2.49059144706e-08
    # 20.8361 14.7415
    # 0.0111931399853
    # 5
    # decoder weight 8.98473663793e-05
    # encoder weight 0.110821345386
    # decoder bias 0.320229262387
    # use new weight
    # decoder weight 2.61451235151e-08
    # encoder weight 2.3984876869e-08
    # decoder bias 2.81146839595e-08
    # 39.3739 21.0313
    # 2.25851527192e-07
    # 6
    # decoder weight 2.96732857843e-08
    # encoder weight 0.22121036619
    # decoder bias 0.568965598334
    # use new weight
    # decoder weight 2.62446303486e-08
    # encoder weight 2.47224072835e-08
    # decoder bias 3.05768801369e-08
    # 5.38292 3.89925
    # 0.00920927685535
    # 7
    # decoder weight 2.75615963256e-06
    # encoder weight 0.0280981070349
    # decoder bias 0.164080616061
    # use new weight
    # decoder weight 2.63295467238e-08
    # encoder weight 2.44789707071e-08
    # decoder bias 2.20199791863e-08
    # 18.2893 11.37
    # 4.22910324294e-07
    # 8
    # decoder weight 3.0026087854e-08
    # encoder weight 0.101342601489
    # decoder bias 0.351288150682
    # use new weight
    # decoder weight 2.66324169701e-08
    # encoder weight 2.43670111255e-08
    # decoder bias 2.67602953118e-08
    # 82.7158 48.0068
    # 2.28855024558e-07
    # 9
    # decoder weight 3.350114275e-08
    # encoder weight 0.476093116138
    # decoder bias 0.773677459927
    # use new weight
    # decoder weight 2.72904499331e-08
    # encoder weight 2.43044516448e-08
    # decoder bias 2.17212788937e-08
    # 45.2502 25.4231
    # 3.64612508628e-07