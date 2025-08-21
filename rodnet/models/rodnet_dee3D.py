from .backbones.deep3D import Deep3Dv1_t, Deep3Dv1_s, Deep3Dv1_m, Deep3Dv1_l
from .modules.mnet import MNet
from torch import nn



class RODNetDeep3DMNet(nn.Module):
    def __init__(self, in_channels, n_class, mnet_cfg=None, train_type='single', model_type='t'):
        super(RODNetDeep3DMNet, self).__init__()
        self.conv_op = nn.Conv3d
        in_chirps_mnet, out_channels_mnet = mnet_cfg
        assert in_channels == in_chirps_mnet
        self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
        if model_type == 't':
            self.net = Deep3Dv1_t(out_channels_mnet, n_class, use_mse_loss=False, train_type=train_type)
        if model_type == 's':
            self.net = Deep3Dv1_s(out_channels_mnet, n_class, use_mse_loss=False, train_type=train_type)
        if model_type == 'm':
            self.net = Deep3Dv1_m(out_channels_mnet, n_class, use_mse_loss=False, train_type=train_type)
        if model_type == 'l':
            self.net = Deep3Dv1_l(out_channels_mnet, n_class, use_mse_loss=False, train_type=train_type)

    def forward(self, x):
        x = self.mnet(x)
        x, pros = self.net(x)
        return x, pros