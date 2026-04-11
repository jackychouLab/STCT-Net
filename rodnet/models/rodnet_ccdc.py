import torch
import torch.nn as nn
from .backbones.complex_basic_unit import ComplexRELU, ComplexStem, ComplexBatchNorm3D, ComplexConv3d, ComplexUpConv3d

class CRadarVanilla(nn.Module):

    def __init__(self, in_channels, n_class, use_mse_loss=False):
        super(CRadarVanilla, self).__init__()
        self.encoder = RODCEncode(in_channels=in_channels)
        self.decoder = RODCDecode(n_class=n_class)
        self.exo_head = nn.Conv3d(in_channels=2 * n_class, out_channels=n_class, kernel_size=3, stride=1, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        exo = x.detach()
        exo = torch.concatenate([exo[..., 0], exo[..., 1]], dim=1)
        exo = self.exo_head(exo)
        if not self.use_mse_loss:
            x[..., 0] = self.sigmoid(x[..., 0])
            x[..., 1] = self.sigmoid(x[..., 1])
            exo = self.sigmoid(exo)
        return torch.concatenate([x[..., 0], x[..., 1], exo], dim=1)


class RODCEncode(nn.Module):

    def __init__(self, in_channels):
        super(RODCEncode, self).__init__()
        self.conv1a = ComplexConv3d(in_channels=in_channels, out_channels=64, kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = ComplexConv3d(in_channels=64, out_channels=64, kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = ComplexConv3d(in_channels=64, out_channels=128, kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = ComplexConv3d(in_channels=128, out_channels=128, kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = ComplexConv3d(in_channels=128, out_channels=256, kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = ComplexConv3d(in_channels=256, out_channels=256, kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = ComplexBatchNorm3D(num_features=64)
        self.bn1b = ComplexBatchNorm3D(num_features=64)
        self.bn2a = ComplexBatchNorm3D(num_features=128)
        self.bn2b = ComplexBatchNorm3D(num_features=128)
        self.bn3a = ComplexBatchNorm3D(num_features=256)
        self.bn3b = ComplexBatchNorm3D(num_features=256)
        self.relu = ComplexRELU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)
        return x


class RODCDecode(nn.Module):

    def __init__(self, n_class):
        super(RODCDecode, self).__init__()
        self.convt1 = ComplexUpConv3d(in_channels=256, out_channels=128, kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = ComplexUpConv3d(in_channels=128, out_channels=64, kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = ComplexUpConv3d(in_channels=64, out_channels=n_class, kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = ComplexBatchNorm3D(num_features=128)
        self.bn2 = ComplexBatchNorm3D(num_features=64)
        self.relu = ComplexRELU()

    def forward(self, x):
        x = self.relu(self.bn1(self.convt1(x)))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.relu(self.bn2(self.convt2(x)))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.convt3(x)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        return x


class RODNetCCDC(nn.Module):
    def __init__(self, in_channels, n_class, mnet_cfg, final_type):
        super(RODNetCCDC, self).__init__()
        in_chirps_mnet, out_channels_mnet = mnet_cfg
        self.stem = ComplexStem(in_channels=in_chirps_mnet, out_channels=out_channels_mnet)
        self.ccdc = CRadarVanilla(out_channels_mnet, n_class, use_mse_loss=False)
        self.final_type = final_type

    def forward(self, x):
        # B real&imag F chirps R A --> B chirps F R A real&imag
        x = x.permute(0, 3, 2, 4, 5, 1).contiguous()
        x = self.stem(x)
        x = self.ccdc(x)
        if self.training is False and self.final_type != "merge":
            xs = x.chunk(3, 1)
            if self.final_type == "real":
                x = torch.cat([xs[1], xs[2], xs[0]], dim=1)
            elif self.final_type == "imag":
                x = torch.cat([xs[0], xs[2], xs[1]], dim=1)
            else:
                raise ValueError("final_type must be 'real' or 'imag' or 'merge'")
        return x