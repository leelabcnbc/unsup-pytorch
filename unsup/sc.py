"""sparse coding. this corresponds to *FistaL1 modules in the original repo"""

from copy import deepcopy
from spams import lasso  # intially I put this line later; and it failed.
from torch.autograd import Variable
from torch import Tensor
import numpy as np
from torch.nn import Linear, MSELoss
from .core import UnsupModule


class LinearSC(UnsupModule):
    def __init__(self, input_size, num_basis, lam,
                 solver_type='spams', solver_params=None,
                 size_average_recon=False,
                 size_average_l1=False):
        super().__init__()
        self.lam = lam
        self.linear_module = Linear(num_basis, input_size, bias=False)
        self.cost = MSELoss(size_average=size_average_recon)
        self.size_average_l1 = size_average_l1
        self.solver_type = solver_type
        self.solver_params = deepcopy(solver_params)

    def forward(self, x: Variable):
        # x is B x input_size matrix to reconstruct
        # init_guess is B x output_size matrix code guess
        if self.solver_type == 'spams':
            x_cpu = x.data.cpu().numpy()
            assert x_cpu.ndim == 2 and x_cpu.shape[1] == self.linear_module.out_features

            # init_guess_shape = (x.shape[0], self.linear_module.in_features)
            # if init_guess is not None:
            #     init_guess = init_guess.data.cpu().numpy()
            # else:
            #     init_guess = np.zeros(init_guess_shape)
            # assert init_guess.shape == init_guess_shape

            # spams does not need any init guess.
            response = lasso(np.asfortranarray(x_cpu.T),
                             D=np.asfortranarray(self.linear_module.weight.data.cpu().numpy()),
                             lambda1=self.lam / 2,  # so to remove 1/2 factor for reconstruction loss.
                             mode=2)
            response = response.T.toarray()  # because lasso returns sparse matrix...
            response_cost = abs(response)*self.lam
            if self.size_average_l1:
                response_cost = float(response_cost.mean())
            else:
                response_cost = float(response_cost.sum())
            # then pass response into self.linear_module
            response = Tensor(response)
            if x.is_cuda:
                response = response.cuda()
            response = Variable(response)

            recon = self.linear_module(response)

            # return reconstruction loss, so that we can do gradient descent on weight later.
            return self.cost(recon, x) + response_cost
        else:
            raise NotImplementedError
