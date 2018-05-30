"""this file tries to replicate https://github.com/koraykv/unsup/blob/master/demo/demo_fista.lua,
with its original experiment parameters.
the original script's data for first 10 iterations were collected using `/debug/reference/demo_fista_debug.lua`
"""
import os
import numpy as np
from numpy.linalg import norm
from unsup import dir_dictionary, sc  # somehow I need to import spams earlier than torch.
from torch import Tensor, optim
from torch.autograd import Variable


def demo(data_dir=dir_dictionary['debug_reference'],
         data_file_prefix='demo_fista_debug'):
    # load all data.
    data_dict = {
        k: np.load(os.path.join(data_dir, f'{data_file_prefix}_{k}.npy')) for k in ('data', 'serr',
                                                                                    'weight', 'grad_weight')
    }
    # print(data_dict['weight'].shape)
    num_data = data_dict['data'].shape[0]
    assert num_data == data_dict['serr'].shape[0] == data_dict['weight'].shape[0] == data_dict['grad_weight'].shape[0]

    model = sc.LinearSC(81, 32, 1.0)
    # intialize.
    init_weight = Tensor(data_dict['weight'][0])
    assert model.linear_module.weight.size() == init_weight.size()
    model.linear_module.weight.data[...] = init_weight
    # change factor from 32 to 81 or 1 will dramatically increase the difference between ref and actual.
    optimizer = optim.SGD(model.parameters(), lr=0.01 / 32)

    # let's have the for loop
    for i, (data_this, weight_this, grad_this, serr_this) in enumerate(
            zip(data_dict['data'], data_dict['weight'], data_dict['grad_weight'], data_dict['serr'])
    ):
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

        cost_this = model.forward(Variable(Tensor(data_this[np.newaxis])))
        cost_this.backward()
        cost_this = cost_this.data.cpu().numpy()
        assert cost_this.shape == (1,)
        cost_this = cost_this[0]
        print(cost_this, serr_this)

        # check grad
        grad_this_actual = model.linear_module.weight.grad.data.cpu().numpy()
        assert grad_this_actual.shape == (81, 32)
        print(norm(grad_this_actual.ravel() - grad_this.ravel()) / (norm(grad_this.ravel()) + 1e-6))

        optimizer.step()


# sample output.
#
#
# 0
# 2.61611268749e-08
# 43.1805 43.1805
# 0.00290196223956
# 1
# 5.35240718639e-06
# 20.2959 20.2959
# 0.000514103349745
# 2
# 5.35213082833e-06
# 36.0171 36.0171
# 0.000500145344171
# 3
# 5.36931814231e-06
# 3.54644 3.54644
# 0.00250252509634
# 4
# 5.3699370032e-06
# 5.21084 5.21084
# 0.0180147354815
# 5
# 5.38090078686e-06
# 3.53002 3.53002
# 0.0
# 6
# 5.38090078686e-06
# 11.3661 11.3661
# 0.00231359645681
# 7
# 5.39583366385e-06
# 40.9056 40.9057
# 0.00178441112682
# 8
# 5.80707565172e-06
# 33.7171 33.7171
# 0.00211863190168
# 9
# 6.23248084483e-06
# 24.2264 24.2264
# 0.00058858690448


if __name__ == '__main__':
    demo()
