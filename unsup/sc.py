"""sparse coding. this corresponds to *FistaL1 modules in the original repo"""

from copy import deepcopy
from functools import partial
from spams import lasso  # intially I put this line later; and it failed.
from torch.autograd import Variable
import torch
from torch import Tensor
import numpy as np
from torch.nn import Linear, MSELoss, ConvTranspose2d
from torch.nn.functional import linear, softshrink, conv2d
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
            cost = torch.abs(x) * self.lam
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
            return self.cost(recon, x) + self.g_fista(response.data), response
        else:
            raise NotImplementedError


def generate_weight_template(h, w, k):
    h_1 = np.arange(1, h + 1)
    w_1 = np.arange(1, w + 1)

    h_1 = np.minimum(h_1, k)
    h_1 = np.minimum(h_1, h_1[::-1])

    w_1 = np.minimum(w_1, k)
    w_1 = np.minimum(w_1, w_1[::-1])

    a = h_1[:, np.newaxis] * w_1[np.newaxis, :]
    return Tensor(a / a.max())


class ConvSC(UnsupModule):
    # here this input_size is the size (height/width) of the image.
    # I assume height = width
    def __init__(self, num_basis, kernel_size, lam,
                 solver_type='fista_custom', solver_params=None,
                 size_average_recon=False,
                 size_average_l1=False, im_size=None, legacy_bias=False):
        super().__init__()

        # TODO: consistently change the semantic of size_average
        # or I always use size_average_* = False and handle this later on.
        self.lam = lam
        self.linear_module = ConvTranspose2d(num_basis, 1, kernel_size, bias=legacy_bias)
        # the official implement has bias.
        if legacy_bias:
            self.linear_module.bias.data.zero_()
        # TODO handle weighted version.
        self.cost = MSELoss(size_average=size_average_recon)
        self.size_average_l1 = size_average_l1
        self.solver_type = solver_type
        self.solver_params = deepcopy(solver_params)
        # save a template for later use.
        assert im_size is not None
        self.register_buffer('_template_weight', Variable(generate_weight_template(im_size,
                                                                                   im_size,
                                                                                   kernel_size)))

        # self.register_buffer('_template_weight', Variable(Tensor(np.ones((25, 25)))))

        # define the function for fista_custom.
        def f_fista(target: Tensor, code: Tensor, calc_grad):
            recon = self.linear_module(Variable(code, volatile=True))
            cost_recon = 0.5 * self.cost(recon, self._template_weight * Variable(target, volatile=True)).data[0]
            if not calc_grad:
                return cost_recon
            else:
                # compute grad.
                grad_this = conv2d(recon - self._template_weight * Variable(target, volatile=True),
                                   self.linear_module.weight, None).data
                return cost_recon, grad_this

        def g_fista(x: Tensor):
            cost = torch.abs(x) * self.lam
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
        n, c, h, w = x.size()
        assert c == 1

        if self.solver_type == 'fista_custom':
            # all assuming stride=1
            xinit = torch.zeros(n, self.linear_module.in_channels,
                                h - self.linear_module.kernel_size[0] + 1,
                                w - self.linear_module.kernel_size[0] + 1)
            if x.is_cuda:
                xinit = xinit.cuda()
            response, L_new = fista_ls(partial(self.f_fista, x.data), self.g_fista, self.pl, xinit,
                                       verbose=False, L=self.L)
            response = Variable(response, requires_grad=False)
            # i think this is kind of speed up trick.
            # in the original lua code,
            # there's another cap on L (0.1), compared to LinearSC.
            if L_new == self.L:
                self.L = max(0.1, L_new / 2)
            else:
                self.L = L_new
            recon = self.linear_module(response)
            return 0.5 * self.cost(recon, self._template_weight * x) + self.g_fista(response.data), response
        else:
            raise NotImplementedError

    def normalize(self):
        # weight is (num_basis, 1, kernel_size, kernel_size).
        # according to
        # https://github.com/koraykv/unsup/blob/54c7a7e0843049436ae3dcd20d9d10716c2ba5cb/FistaL1.lua#L188-L192
        # in the original code, weight matrix is 3d because num_basis x 1 is mixed together.

        num_basis, _, k_1, k_2 = self.linear_module.weight.size()
        assert _ == 1
        # then view its `data` another way
        weight_reshaped = self.linear_module.weight.data.view(num_basis, k_1 * k_2)
        # since this is a view, the original one will be modified.
        weight_reshaped.div_(torch.norm(weight_reshaped, 2, 1, keepdim=True) + 1e-12)
