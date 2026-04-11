from .backbones.complexDeep3D import ComplexNet_t, ComplexNet_s
from torch import nn


class RODNetDeep3DComplex(nn.Module):
    def __init__(self, in_channels, n_class, train_type='single', model_type='t'):
        super(RODNetDeep3DComplex, self).__init__()
        if model_type == 't':
            self.net = ComplexNet_t(in_channels, n_class, use_mse_loss=False, train_type=train_type)
        if model_type == 's':
            self.net = ComplexNet_s(in_channels, n_class, use_mse_loss=False, train_type=train_type)

    def forward(self, x):
        x, pros = self.net(x)
        return x, pros