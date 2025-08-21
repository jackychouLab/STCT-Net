import torch.nn as nn

from .backbones.hg import RadarStackedHourglass
from .modules.mnet import MNet

try:
    from ..ops.dcn import DeformConvPack3D
except:
    print("Warning: DCN modules are not correctly imported!")


class RODNetHGDCN(nn.Module):
    def __init__(self, in_channels, n_class, stacked_num=2, mnet_cfg=None, dcn=True):
        super(RODNetHGDCN, self).__init__()
        self.dcn = dcn
        if dcn:
            self.conv_op = DeformConvPack3D
        else:
            self.conv_op = nn.Conv3d
        if mnet_cfg is not None:
            in_chirps_mnet, out_channels_mnet = mnet_cfg
            assert in_channels == in_chirps_mnet
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
            self.with_mnet = True
            self.stacked_hourglass = RadarStackedHourglass(out_channels_mnet, n_class, stacked_num=stacked_num,
                                                           conv_op=self.conv_op)
        else:
            self.with_mnet = False
            self.stacked_hourglass = RadarStackedHourglass(in_channels, n_class, stacked_num=stacked_num,
                                                           conv_op=self.conv_op)

    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        out = self.stacked_hourglass(x)
        return out