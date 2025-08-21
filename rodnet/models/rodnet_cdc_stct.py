import torch.nn as nn

from .backbones.cdc import RadarVanilla
from .rodnet_myNet import MNetv5 as MNet

try:
    from ..ops.dcn import DeformConvPack3D
except:
    print("Warning: DCN modules are not correctly imported!")


class RODNetCDCDCNSTCT(nn.Module):
    def __init__(self, in_channels, n_class, mnet_cfg=None):
        super(RODNetCDCDCNSTCT, self).__init__()
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            assert in_channels == in_chirps_mnet
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet)
            self.with_mnet = True
            self.cdc = RadarVanilla(out_channels_mnet, n_class, use_mse_loss=False)
        else:
            self.with_mnet = False
            self.cdc = RadarVanilla(in_channels, n_class, use_mse_loss=False)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        x = self.cdc(x)
        return x
