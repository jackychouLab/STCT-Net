import torch
import torch.nn as nn
from timm.models.layers import to_2tuple



class Linear(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.linear = nn.Linear(in_dims, out_dims, bias=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class SepConv(nn.Module):
    def __init__(self, dim, act, full_conv, expansion_ratio=2, kernel_size=(3, 7, 7), padding=(1, 3, 3)):
        super(SepConv, self).__init__()
        med_channels = int(expansion_ratio * dim)
        if full_conv:
            self.pwconv1 = nn.Conv3d(dim, med_channels, 1, 1, 0, bias=True)
        else:
            self.pwconv1 = Linear(dim, med_channels)
        self.act = act()
        self.dwconv = nn.Conv3d(med_channels, med_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=med_channels, bias=True)
        if full_conv:
            self.pwconv2 = nn.Conv3d(med_channels, dim, 1, 1, 0, bias=True)
        else:
            self.pwconv2 = Linear(med_channels, dim)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, act, full_conv, mlp_ratio=4, drop=0.1):
        super(Mlp, self).__init__()
        in_features = dim
        out_features = in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)
        if full_conv:
            self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1, 0, bias=True)
        else:
            self.fc1 = Linear(in_features, hidden_features)
        self.act = act()
        self.drop1 = nn.Dropout(drop_probs[0])
        if full_conv:
            self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1, 0, bias=True)
        else:
            self.fc2 = Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MetaFormerBlock(nn.Module):
    def __init__(self, dim, norm, act, full_conv, token_mixer=SepConv, mlp=Mlp):
        super(MetaFormerBlock, self).__init__()
        self.norm1 = norm(dim // 4, dim, 1e-6)
        self.token_mixer = token_mixer(dim, act, full_conv)
        self.norm2 = norm(dim // 4, dim, 1e-6)
        self.mlp = mlp(dim, act, full_conv)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RODEncode(nn.Module):
    def __init__(self, in_channels, norm, act, full_conv, depths=[2, 2, 4]):
        super(RODEncode, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2), bias=True),
            norm(16, 64, 1e-6),
        )

        self.encoder_block1 = nn.Sequential(
            *[MetaFormerBlock(64, norm, act, full_conv)
              for _ in range(depths[0])])

        self.dowm_sample1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            norm(16, 64, 1e-6),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            act()
        )

        self.encoder_block2 = nn.Sequential(
            *[MetaFormerBlock(128, norm, act, full_conv)
              for _ in range(depths[1])])

        self.down_sample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            norm(32, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            act()
        )

        self.encoder_block3 = nn.Sequential(
            *[MetaFormerBlock(256, norm, act, full_conv)
              for _ in range(depths[2])])

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder_block1(x)
        x1 = x
        x = self.dowm_sample1(x)
        x = self.encoder_block2(x)
        x2 = x
        x = self.down_sample2(x)
        x = self.encoder_block3(x)
        return x, x1, x2


class RODDecode(nn.Module):
    def __init__(self, n_class, head_size, norm, act, full_conv, depths=[2, 2]):
        super(RODDecode, self).__init__()
        self.up_sample1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            norm(64, 256, 1e-6),
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            act()
        )
        self.merge1 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            norm(32, 128, 1e-6),
            act()
        )
        self.decoder_block1 = nn.Sequential(
            *[MetaFormerBlock(128, norm, act, full_conv)
              for _ in range(depths[0])])

        self.up_sample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            norm(32, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            act()
        )
        self.merge2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            norm(16, 64, 1e-6),
            act()
        )
        self.decoder_block2 = nn.Sequential(
            *[MetaFormerBlock(64, norm, act, full_conv)
              for _ in range(depths[1])])

        self.up_sample3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
        )
        self.head = nn.Sequential(
            norm(16, 64, 1e-6),
            nn.Conv3d(in_channels=64, out_channels=n_class, kernel_size=head_size, stride=1, padding=((head_size[0] - 1) // 2, (head_size[1] - 1) // 2, (head_size[2] - 1) // 2), bias=True),
        )

    def forward(self, x, x1, x2):
        x = self.up_sample1(x)
        x = self.merge1(torch.cat([x, x2], dim=1))
        out1 = self.decoder_block1(x)
        x = self.up_sample2(out1)
        x = self.merge2(torch.cat([x, x1], dim=1))
        out2 = self.decoder_block2(x)
        out3 = self.up_sample3(out2)
        out3 = self.head(out3)
        return out3


class STCTM(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(STCTM, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.d_net = nn.Sequential(
            nn.Conv2d(in_chirps, out_channels // 2, 1, 1, 0, bias=True),
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.GELU()
        )
        self.s_net = nn.Sequential(
            nn.Conv3d(2, out_channels // 2, (1, 3, 3), 1, (0, 1, 1), bias=True),
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.GELU(),
            nn.AvgPool3d((in_chirps, 1, 1))
        )
        self.m_net = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = x[:, :, win, :, :, :]
            x_c = x_win[:, 0, ...] + 1j * x_win[0, 1, ...]
            x_mag = torch.abs(torch.fft.fft(x_c, dim=1))
            x_d = self.d_net(x_mag)
            x_s = self.s_net(x_win).squeeze(2)
            x_out[:, :, win, :, :] = self.m_net(torch.cat([x_d, x_s], dim=1))
        return x_out


class STCTNet(nn.Module):
    def __init__(self, mnet_cfg, n_class):
        super(STCTNet, self).__init__()
        head_size = (3, 3, 3)
        full_conv = True
        norm = nn.GroupNorm
        act = nn.GELU
        self.mnet = STCTM(in_chirps=mnet_cfg[0], out_channels=mnet_cfg[1])
        self.encoder = RODEncode(mnet_cfg[1], norm, act, full_conv)
        self.decoder = RODDecode(n_class, head_size, norm, act, full_conv)

    def forward(self, x):
        x = self.mnet(x)
        x, x1, x2 = self.encoder(x)
        x = self.decoder(x, x1, x2)
        return x