"""implements *psd.lua in the original code"""

from torch.nn import MSELoss, Parameter, Sequential, Tanh, Linear, Module
from torch import ones
from .core import UnsupModule
from .sc import LinearSC


class PSD(UnsupModule):
    def __init__(self, encoder, decoder,
                 beta,  # I personally feel beta is irrelevant for learning,
                 ):
        super().__init__()
        self.beta = beta
        self.encoder = encoder  # 1 NN layer.
        self.decoder = decoder  # FISTA L1, SC
        # I will only support size_average = False for now.
        self.predcost = MSELoss(size_average=False)

    def forward(self, x):
        # go through encoder
        code_forward = self.encoder(x)
        # do sparse coding.
        # TODO: add code_forward as initialization?
        # in lua Torch, this is actually useless,
        # as optim.fistals is not implemented correctly
        # to handle init value.
        sc_cost, sc_code = self.decoder(x)
        pred_cost = self.predcost(code_forward, sc_code)
        # print('sc cost', sc_cost)
        # print('pred_cost', pred_cost)
        return sc_cost + self.beta * pred_cost


# scaling layer
# https://discuss.pytorch.org/t/is-scale-layer-available-in-pytorch/7954/8
# implements Diag.lua in the original code.
class ScaleLayer(Module):
    def __init__(self, shape, init_val=1.0):
        super().__init__()
        self.scale = Parameter(ones(*shape) * init_val)

    def forward(self, x):
        return x * self.scale


class LinearPSD(PSD):
    def __init__(self, input_size, num_basis, lam, beta):
        # currently, pure linear; handler other cases later.
        super().__init__(Sequential(Linear(input_size, num_basis),
                                    # Tanh(),
                                    # ScaleLayer((num_basis,))
                                    ),
                         LinearSC(input_size, num_basis, lam, solver_type='fista_custom'),
                         beta)
