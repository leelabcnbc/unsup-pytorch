"""sparse coding. this corresponds to *FistaL1 modules in the original repo"""

from copy import deepcopy
from functools import partial
from spams import lasso  # intially I put this line later; and it failed.
from torch.autograd import Variable
import torch
from torch import Tensor
import numpy as np
from torch.nn import Linear, MSELoss
from torch.nn.functional import linear, softshrink
from .core import UnsupModule
from .fista import fista_ls


# check http://bmihub.org/project/stackedpsd
# for a proper minimization of both sparse coding loss AND prediction loss.

class LinearSC(UnsupModule):
    def __init__(self, input_size, num_basis, lam,
                 solver_type='spams', solver_params=None,
                 size_average_recon=False,
                 size_average_l1=False):
        super().__init__()

        # TODO: consistently change the semantic of size_average
        # or I always use size_average_* = False and handle this later on.
        self.lam = lam
        self.linear_module = Linear(num_basis, input_size, bias=False)
        self.cost = MSELoss(size_average=size_average_recon)
        self.size_average_l1 = size_average_l1
        self.solver_type = solver_type
        self.solver_params = deepcopy(solver_params)

        # define the function for fista_custom.
        def f_fista(target: Tensor, code: Tensor, calc_grad):
            recon = self.linear_module(Variable(code, volatile=True))
            cost_recon = self.cost(recon, Variable(target, volatile=True)).data[0]
            if not calc_grad:
                return cost_recon
            else:
                # compute grad.
                grad_this = linear(recon - Variable(target, volatile=True),
                                   self.linear_module.weight.t()).data
                grad_this *= 2
                return cost_recon, grad_this

        def g_fista(x: Tensor):
            cost = torch.abs(x)*self.lam
            if self.size_average_l1:
                cost = cost.mean()
            else:
                cost = cost.sum()
            return cost

        self.f_fista = f_fista
        self.g_fista = g_fista
        self.pl = lambda x, L: softshrink(Variable(x, volatile=True), lambd=self.lam / L).data

        if self.solver_type == 'fista_custom':
            self.L = 0.1  # init L

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
            response_cost = abs(response) * self.lam
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
        elif self.solver_type == 'fista_custom':
            # using my hand written fista code, which should reproduce the lua code exactly (up to machine precision).
            xinit = torch.zeros(x.size()[0], self.linear_module.in_features)
            if x.is_cuda:
                xinit = xinit.cuda()
            response, L_new = fista_ls(partial(self.f_fista, x.data), self.g_fista, self.pl, xinit,
                                       verbose=False, L=self.L)
            response = Variable(response, requires_grad=False)
            # i think this is kind of speed up trick.
            # TODO: in FistaL1.lua (not LinearFistaL1.lua),
            # there's another cap on L.
            if L_new == self.L:
                self.L = L_new / 2
            else:
                self.L = L_new
            recon = self.linear_module(response)
            return self.cost(recon, x) + self.g_fista(response.data)
        else:
            raise NotImplementedError
