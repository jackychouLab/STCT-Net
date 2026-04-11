import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath



class ContextBranch(nn.Module):

    def __init__(self, in_channels, depths=[3, 3, 9, 3]):
        super(ContextBranch, self).__init__()
        dp_rates = [x.item() for x in torch.linspace(0, 0.2, sum(depths))]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.GroupNorm(64 // 8, 64, 1e-6),
        )

        self.encoder_block1 = nn.Sequential(
            *[MetaFormerBlock(64, drop_path)
              for _, drop_path in zip(range(depths[0]), dp_rates[:3])])

        self.dowm_sample1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.GroupNorm(64 // 8, 64, 1e-6),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block2 = nn.Sequential(
            *[MetaFormerBlock(128, drop_path)
              for _, drop_path in zip(range(depths[1]), dp_rates[3:6])])

        self.down_sample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(128 // 8, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block3 = nn.Sequential(
            *[MetaFormerBlock(256, drop_path)
              for _, drop_path in zip(range(depths[2]), dp_rates[6:15])])

        self.down_sample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(256 // 8, 256, 1e-6),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block4 = nn.Sequential(
            *[MetaFormerBlock(512, drop_path)
              for _, drop_path in zip(range(depths[3]), dp_rates[15:18])])

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder_block1(x)

        x = self.dowm_sample1(x)
        x = self.encoder_block2(x)
        x2 = x

        x = self.down_sample2(x)
        x = self.encoder_block3(x)
        x3 = x

        x = self.down_sample3(x)
        x = self.encoder_block4(x)
        return x, x3, x2


class SepConv(nn.Module):

    def __init__(self, dim, expansion_ratio=2, kernel_size=(3, 7, 7), padding=(1, 3, 3)):
        super(SepConv, self).__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv3d(dim, med_channels, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.dwconv = nn.Conv3d(med_channels, med_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=med_channels, bias=True)
        self.pwconv2 = nn.Conv3d(med_channels, dim, 1, 1, 0, bias=True)


    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        return x


class Mlp(nn.Module):

    def __init__(self, dim, mlp_ratio=2):
        super(Mlp, self).__init__()
        in_features = dim
        out_features = in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1, 0, bias=True)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):

    def __init__(self, dim, drop_path=0., token_mixer=SepConv, mlp=Mlp):
        super(MetaFormerBlock, self).__init__()

        self.norm1 = nn.GroupNorm(dim // 8, dim, 1e-6)
        self.token_mixer = token_mixer(dim)
        self.norm2 = nn.GroupNorm(dim // 8, dim, 1e-6)
        self.mlp = mlp(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class MNet(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNet, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels

        self.s_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.GroupNorm(out_channels // 2 // 8, out_channels // 2),
            nn.SiLU(inplace=True),
            nn.AvgPool3d(kernel_size=(in_chirps, 1, 1)),
        )
        self.t_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=True),
            nn.GroupNorm(out_channels // 2 // 8, out_channels // 2),
            nn.SiLU(inplace=True),
            nn.AvgPool3d(kernel_size=(in_chirps, 1, 1)),
        )

        self.m_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True),
            nn.GroupNorm(out_channels // 8, out_channels),
            nn.SiLU(inplace=True),
        )


    def forward(self, x):
        b, c, win, chirps, h, w = x.shape
        x = x.transpose(1, 2).reshape(-1, c, chirps, h, w).contiguous()
        x_t = self.t_conv3d(x).squeeze(2)
        x_s = self.s_conv3d(x).squeeze(2)
        x_m = torch.cat([x_t, x_s], dim=1).squeeze(2).reshape(b, win, -1, h, w).transpose(1, 2).contiguous()
        out = self.m_conv3d(x_m)
        return out


class DBNet_base(nn.Module):
    def __init__(self, mnet_cfg, n_class, train_type):
        super(DBNet_base, self).__init__()
        assert train_type in ['multi']
        self.mnet = MNet(in_chirps=mnet_cfg[0], out_channels=mnet_cfg[1])
        self.contextBranch = ContextBranch(mnet_cfg[1])
        self.f4_fpn_conv = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(128 // 8, 128, eps=1e-6),
            nn.SiLU(inplace=True),
        )
        self.f3_fpn_conv = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(128 // 8, 128, eps=1e-6),
            nn.SiLU(inplace=True),
        )
        self.f2_fpn_conv = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(128 // 8, 128, eps=1e-6),
            nn.SiLU(inplace=True),
        )

        self.f3_refine_conv = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, groups=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.GroupNorm(128 // 8, 128, eps=1e-6),
            nn.SiLU(inplace=True),
        )
        self.f2_refine_conv = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.GroupNorm(128 // 8, 128, eps=1e-6),
            nn.SiLU(inplace=True),
        )

        self.f1_time_conv = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.GroupNorm(128 // 8, 128, eps=1e-6),
            nn.SiLU(inplace=True),
        )
        self.f1_refine_conv = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, bias=False),
            nn.GroupNorm(128 // 8, 128, eps=1e-6),
            nn.SiLU(inplace=True),
        )
        self.scale = 0.1

        self.f1_head = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(64 // 8, 64, eps=1e-6),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(64 // 8, 64, eps=1e-6),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, n_class, kernel_size=1, bias=True),
        )

        self.f2_head = nn.Sequential(
            nn.Conv3d(128, n_class, kernel_size=1, bias=True),
        )

        self.f3_head = nn.Sequential(
            nn.Conv3d(128, n_class, kernel_size=1, bias=True),
        )

    def forward(self, x):
        x = self.mnet(x)

        f4, f3, f2 = self.contextBranch(x)
        f4 = self.f4_fpn_conv(f4)
        f3 = self.f3_fpn_conv(f3)
        f2 = self.f2_fpn_conv(f2)

        f4 = F.interpolate(f4, scale_factor=(1, 2, 2), mode='nearest')
        f3 = f3 + f4
        f3_out = self.scale * self.f3_refine_conv(f3) + f3

        f3 = F.interpolate(f3_out, scale_factor=(1, 2, 2), mode='nearest')
        f2 = f3 + f2
        f2_out = self.scale * self.f2_refine_conv(f2) + f2

        f2 = F.interpolate(f2_out, scale_factor=(2, 1, 1), mode='nearest')
        f2 = self.scale * self.f1_time_conv(f2) + f2
        f2 = F.interpolate(f2, scale_factor=(1, 2, 2), mode='nearest')
        f1_out = self.scale * self.f1_refine_conv(f2) + f2
        f1_out = self.f1_head(f1_out)
        if self.training:
            f3_out = self.f3_head(f3_out)
            f2_out = self.f2_head(f2_out)
            return [f3_out, f2_out, f1_out]
        else:
            b_out, c_out, f_out, r_out, a_out = f1_out.shape
            f1_out = f1_out.permute(0, 2, 1, 3, 4).reshape(-1, c_out, r_out, a_out)
            f1_out = F.interpolate(f1_out, scale_factor=(1, 2, 2), mode='bilinear', align_corners=True).reshape(b_out, f_out, c_out, int(r_out * 2), int(a_out * 2)).permute(0, 2, 1, 3, 4)
            return f1_out
