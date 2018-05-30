from torch.nn import Module


class UnsupModule(Module):
    def __init__(self):
        super().__init__()

    def normalize(self):
        raise NotImplementedError

    def forward(self, *input_this):
        raise NotImplementedError
