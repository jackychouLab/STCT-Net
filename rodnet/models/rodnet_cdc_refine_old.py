import torch.nn as nn
from .modules.mnet import MNet
import torch
try:
    from ..ops.dcn import DeformConvPack3D
except:
    print("Warning: DCN modules are not correctly imported!")
import math




class RadarVanilla(nn.Module):

    def __init__(self, in_channels, n_class, use_mse_loss=False, upsample_type='transpose', use_skip=False):
        super(RadarVanilla, self).__init__()
        self.encoder = RODEncode(in_channels=in_channels)
        self.decoder = RODDecode(n_class=n_class, upsample_type=upsample_type, use_skip=use_skip)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if not self.use_mse_loss:
            x = self.sigmoid(x)
        return x


class RODEncode(nn.Module):

    def __init__(self, in_channels=2):
        super(RODEncode, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=in_channels, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x1 = self.relu(self.bn1b(self.conv1b(x1)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x2 = self.relu(self.bn2a(self.conv2a(x1)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x2 = self.relu(self.bn2b(self.conv2b(x2)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x3 = self.relu(self.bn3a(self.conv3a(x2)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x3 = self.relu(self.bn3b(self.conv3b(x3)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)
        return [x1, x2, x3]


class RODDecode(nn.Module):

    def __init__(self, n_class, upsample_type, use_skip):
        super(RODDecode, self).__init__()
        if upsample_type == 'transpose':
            self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2), bias=True)
            self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2), bias=True)
            self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=n_class, kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2), bias=True)
            print('use_transpose')
        elif upsample_type == 'nearest':
            self.convt1 = nn.Sequential(
                nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
                nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), bias=True)
            )
            self.convt2 = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
                nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), bias=True)
            )
            self.convt3 = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
                nn.Conv3d(in_channels=64, out_channels=n_class, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), bias=True)
            )
            print('use_nearest')
        elif upsample_type == 'converse':
            self.convt1 = nn.Sequential(
                Converse3D(256, scale=(1, 2, 2)),
                nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 5, 5), stride=(1, 1, 1),padding=(1, 2, 2), bias=True)
            )
            self.convt2 = nn.Sequential(
                Converse3D(128, scale=(2, 2, 2)),
                nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 5, 5), stride=(1, 1, 1),padding=(1, 2, 2), bias=True)
            )
            self.convt3 = nn.Sequential(
                Converse3D(64, scale=(2, 2, 2)),
                nn.Conv3d(in_channels=64, out_channels=n_class, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), bias=True)
            )
            print('use_converse')
        elif upsample_type == 'converseT':
            self.convt1 = nn.Sequential(
                Converse3DT(256, scale=(1, 2, 2)),
                nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 5, 5), stride=(1, 1, 1),padding=(1, 2, 2), bias=True)
            )
            self.convt2 = nn.Sequential(
                Converse3DT(128, scale=(2, 2, 2)),
                nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 5, 5), stride=(1, 1, 1),padding=(1, 2, 2), bias=True)
            )
            self.convt3 = nn.Sequential(
                Converse3DT(64, scale=(2, 2, 2)),
                nn.Conv3d(in_channels=64, out_channels=n_class, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), bias=True)
            )
            print('use_converse')
        self.prelu = nn.PReLU()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.use_skip = use_skip
        if self.use_skip:
            self.merge_conv1 = nn.Sequential(
                nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(128),
                nn.SiLU()
            )
            self.merge_conv2 = nn.Sequential(
                nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm3d(64),
                nn.SiLU()
            )

    def forward(self, x):
        x1, x2, x3 = x
        if self.use_skip:
            x = self.silu(self.merge_conv1(torch.cat([self.convt1(x3), x2], dim=1)))
            x = self.silu(self.merge_conv2(torch.cat([self.convt2(x), x1], dim=1)))
        else:
            x = self.prelu(self.convt1(x3))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
            x = self.prelu(self.convt2(x))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.convt3(x)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        return x


class RODNetCDC_refine(nn.Module):
    def __init__(self, in_channels, n_class, upsample_type, use_skip, mnet_cfg=None):
        super(RODNetCDC_refine, self).__init__()

        self.conv_op = nn.Conv3d
        in_chirps_mnet, out_channels_mnet = mnet_cfg
        self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=self.conv_op)
        self.with_mnet = True
        self.cdc = RadarVanilla(out_channels_mnet, n_class, use_mse_loss=False, upsample_type=upsample_type, use_skip=use_skip)


    def forward(self, x):
        if self.with_mnet:
            x = self.mnet(x)
        x = self.cdc(x)
        return x


class MNet(nn.Module):
    def __init__(self, in_chirps, out_channels, conv_op=None):
        super(MNet, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        if conv_op is None:
            conv_op = nn.Conv3d
        self.conv_op = conv_op

        self.t_conv3d = conv_op(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                padding=(1, 0, 0))
        t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        self.t_maxpool = nn.MaxPool3d(kernel_size=(t_conv_out, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = self.t_conv3d(x[:, :, win, :, :, :])
            x_win = self.t_maxpool(x_win)
            x_win = x_win.view(batch_size, self.out_channels, w, h)
            x_out[:, :, win, ] = x_win
        return x_out


class Converse3D(nn.Module):
    def __init__(self, in_channels,  kernel_size=3, scale=(2, 2, 2), padding_mode='circular', eps=1e-5):
        super(Converse3D, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = kernel_size - 1
        self.padding_mode = padding_mode
        self.eps = eps

        # ensure depthwise
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1, self.in_channels, -1), dim=-1).view(1, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)


    def forward(self, x):
        exo_x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')

        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0)

        self.biaseps = torch.sigmoid(self.bias - 9.0) + self.eps
        _, _, d, h, w = x.shape
        STy = self.upsample(x, scale=self.scale)

        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')

        FB = self.p2o(self.weight, (d * self.scale[0], h * self.scale[1], w * self.scale[2]))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-3, -2, -1))
        FR = FBFy + torch.fft.fftn(self.biaseps * x, dim=(-3, -2, -1))
        x1 = FB * FR
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR = FBR / (invW + self.biaseps)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale[0], self.scale[1], self.scale[2])
        FX = (FR - FCBinvWBR) / self.biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-3, -2, -1)))

        if self.padding > 0:
            out = out[..., self.padding * self.scale[0]:-self.padding * self.scale[0], self.padding * self.scale[1]:-self.padding * self.scale[1], self.padding * self.scale[2]:-self.padding * self.scale[2]]

        out = torch.add(out, exo_x)
        return out

    def splits(self, a, scale):
        # B C D W H
        *leading_dims, D, W, H = a.size()
        D_s, W_s, H_s = D // scale[0], W // scale[1], H // scale[2]
        # B C 2 D 2 W 2 H
        b = a.view(*leading_dims, scale[0], D_s, scale[1], W_s, scale[2], H_s)
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3, len(leading_dims) + 5, len(leading_dims), len(leading_dims) + 2, len(leading_dims) + 4]
        b = b.permute(*permute_order).contiguous()
        b = b.view(*leading_dims, D_s, W_s, H_s, scale[0] * scale[1] * scale[2])
        return b

    def p2o(self, psf, shape):
        otf = torch.zeros(psf.shape[:-3] + shape).type_as(psf)
        otf[..., :psf.shape[-3], :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-3] / 2), -int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-3, -2, -1))
        otf = torch.fft.fftn(otf, dim=(-3, -2, -1))

        return otf

    def upsample(self, x, scale=(2, 2, 2)):
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale[0], x.shape[3] * scale[1], x.shape[4] * scale[2])).type_as(x)
        z[..., st::scale[0], st::scale[1], st::scale[2]].copy_(x)
        return z


class Converse3DT(nn.Module):
    def __init__(self, in_channels,  kernel_size=3, scale=(2, 2, 2), padding_mode='circular', eps=1e-5):
        super(Converse3DT, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = kernel_size - 1
        self.padding_mode = padding_mode
        self.eps = eps

        # ensure depthwise
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1, self.in_channels, -1), dim=-1).view(1, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)
        if scale[0] == 1:
            self.exo_up = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 6, 6),
                                             stride=(1, 2, 2), padding=(1, 2, 2), bias=True)
        if scale[0] == 2:
            self.exo_up = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(4, 6, 6),
                                             stride=(2, 2, 2), padding=(1, 2, 2), bias=True)

    def forward(self, x):
        exo_x = self.exo_up(x)

        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0)

        self.biaseps = torch.sigmoid(self.bias - 9.0) + self.eps
        _, _, d, h, w = x.shape
        STy = self.upsample(x, scale=self.scale)

        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')

        FB = self.p2o(self.weight, (d * self.scale[0], h * self.scale[1], w * self.scale[2]))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-3, -2, -1))
        FR = FBFy + torch.fft.fftn(self.biaseps * x, dim=(-3, -2, -1))
        x1 = FB * FR
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR = FBR / (invW + self.biaseps)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale[0], self.scale[1], self.scale[2])
        FX = (FR - FCBinvWBR) / self.biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-3, -2, -1)))

        if self.padding > 0:
            out = out[..., self.padding * self.scale[0]:-self.padding * self.scale[0], self.padding * self.scale[1]:-self.padding * self.scale[1], self.padding * self.scale[2]:-self.padding * self.scale[2]]

        out = torch.add(out, exo_x)
        return out

    def splits(self, a, scale):
        # B C D W H
        *leading_dims, D, W, H = a.size()
        D_s, W_s, H_s = D // scale[0], W // scale[1], H // scale[2]
        # B C 2 D 2 W 2 H
        b = a.view(*leading_dims, scale[0], D_s, scale[1], W_s, scale[2], H_s)
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3, len(leading_dims) + 5, len(leading_dims), len(leading_dims) + 2, len(leading_dims) + 4]
        b = b.permute(*permute_order).contiguous()
        b = b.view(*leading_dims, D_s, W_s, H_s, scale[0] * scale[1] * scale[2])
        return b

    def p2o(self, psf, shape):
        otf = torch.zeros(psf.shape[:-3] + shape).type_as(psf)
        otf[..., :psf.shape[-3], :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-3] / 2), -int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-3, -2, -1))
        otf = torch.fft.fftn(otf, dim=(-3, -2, -1))

        return otf

    def upsample(self, x, scale=(2, 2, 2)):
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale[0], x.shape[3] * scale[1], x.shape[4] * scale[2])).type_as(x)
        z[..., st::scale[0], st::scale[1], st::scale[2]].copy_(x)
        return z