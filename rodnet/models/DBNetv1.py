import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath



class ARM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ARM, self).__init__()
        self.feat_conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(out_channel // 4, out_channel, eps=1e-6),
            nn.SiLU(inplace=True),
        )

        self.se_atten = nn.Sequential(
            nn.Conv3d(out_channel, out_channel // 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channel // 4, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.feat_conv(x)
        atten = self.se_atten(F.avg_pool3d(feat, feat.shape[-3:]))
        out = torch.mul(feat, atten)
        return out


class FFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FFM, self).__init__()
        self.feat_conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(out_channel // 4, out_channel, eps=1e-6),
            nn.SiLU(inplace=True),
        )

        self.se_atten = nn.Sequential(
            nn.Conv3d(out_channel, out_channel // 4, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channel // 4, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, fsp, fcp):
        feat = self.feat_conv(torch.cat([fsp, fcp], dim=1))
        atten = self.se_atten(F.avg_pool3d(feat, feat.shape[-3:]))
        feat_atten = torch.mul(feat, atten)
        feat_out = torch.add(feat, feat_atten)
        return feat_out


class ContextBranch(nn.Module):

    def __init__(self, in_channels, depths=[3, 3, 9, 3]):
        super(ContextBranch, self).__init__()
        dp_rates = [x.item() for x in torch.linspace(0, 0.2, sum(depths))]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2), bias=False),
            nn.GroupNorm(64 // 4, 64, 1e-6),
        )

        self.encoder_block1 = nn.Sequential(
            *[MetaFormerBlock(64, drop_path)
              for _, drop_path in zip(range(depths[0]), dp_rates[:3])])

        self.dowm_sample1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.GroupNorm(64 // 4, 64, 1e-6),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block2 = nn.Sequential(
            *[MetaFormerBlock(128, drop_path)
              for _, drop_path in zip(range(depths[1]), dp_rates[3:6])])

        self.down_sample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(128 // 4, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block3 = nn.Sequential(
            *[MetaFormerBlock(256, drop_path)
              for _, drop_path in zip(range(depths[1]), dp_rates[6:15])])

        self.down_sample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(256 // 4, 256, 1e-6),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block4 = nn.Sequential(
            *[MetaFormerBlock(512, drop_path)
              for _, drop_path in zip(range(depths[1]), dp_rates[15:18])])

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder_block1(x1)

        x2 = self.dowm_sample1(x1)
        x2 = self.encoder_block2(x2)

        x3 = self.down_sample2(x2)
        x3 = self.encoder_block3(x3)

        x4 = self.down_sample3(x3)
        x4 = self.encoder_block4(x4)
        return x4, x3, x2, x1


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

        self.norm1 = nn.GroupNorm(dim // 4, dim, 1e-6)
        self.token_mixer = token_mixer(dim)
        self.norm2 = nn.GroupNorm(dim // 4, dim, 1e-6)
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
        self.conv_op = nn.Conv3d
        self.t_conv3d = nn.Conv3d(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=True)
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


class DBNetv1(nn.Module):
    def __init__(self, mnet_cfg, n_class, train_type):
        super(DBNetv1, self).__init__()
        assert train_type in ['multi']
        self.mnet = MNet(in_chirps=mnet_cfg[0], out_channels=mnet_cfg[1])
        self.contextBranch = ContextBranch(mnet_cfg[1])
        self.spatioBranch = nn.Sequential(
            nn.Conv2d(32, 64, 7, 2, 3, bias=False),
            nn.GroupNorm(64 // 4, 64, eps=1e-6),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.GroupNorm(64 // 4, 64, eps=1e-6),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.GroupNorm(64 // 4, 64, eps=1e-6),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 1, 1, 0, bias=True),
        )
        self.avg_conv = nn.Sequential(
            nn.Conv3d(512, 128, 1, 1, 0, bias=False),
            nn.GroupNorm(128 // 4, 128, eps=1e-06),
            nn.SiLU(inplace=True),
        )
        self.f4_arm = ARM(512, 128)
        self.f3_arm = ARM(256, 128)
        self.f4_refine = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1), mode='nearest'),
            nn.GroupNorm(128 // 4, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=(2, 1, 1), mode='nearest'),
            nn.GroupNorm(128 // 4, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(128 // 4, 128, eps=1e-06),
            nn.SiLU(inplace=True),
        )
        self.f3_refine = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1), mode='nearest'),
            nn.GroupNorm(128 // 4, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=(2, 1, 1), mode='nearest'),
            nn.GroupNorm(128 // 4, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(128 // 4, 128, eps=1e-06),
            nn.SiLU(inplace=True),
        )
        self.FFM = FFM(256, 128)
        self.fs_head = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(128 // 4, 128, eps=1e-06),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels=128, out_channels=n_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.f3_head = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(128 // 4, 128, eps=1e-06),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels=128, out_channels=n_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.f4_head = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(128 // 4, 128, eps=1e-06),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels=128, out_channels=n_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mnet(x)

        # Context Information

        f4, f3, f2, f1 = self.contextBranch(x)

        avg = f4.mean(dim=(2, 3, 4), keepdim=True)
        avg = self.avg_conv(avg)
        avg_up = F.interpolate(avg, f4.size()[-3:], mode='nearest')

        f4 = self.f4_arm(f4)
        f4 = avg_up + f4
        f4_up = F.interpolate(f4, f3.size()[-3:], mode='nearest')
        f4_refine = self.f4_refine(f4_up)

        f3 = self.f3_arm(f3)
        f3 = f4_up + f3
        f3_up = F.interpolate(f3, f2.size()[-3:], mode='nearest')
        f3_refine = self.f3_refine(f3_up)

        # Spatio Information
        b, c, f, r, a = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, r, a)
        fs = self.spatioBranch(x)
        _, c_out, r_out, a_out = fs.shape
        fs = fs.reshape(b, f, c_out, r_out, a_out).permute(0, 2, 1, 3, 4)

        # FFM
        fs = self.FFM(fs, f3_refine)
        fs_out = self.fs_head(fs)
        f3_out = self.f3_head(f3_refine)
        f4_out = self.f4_head(f4_refine)

        b_out, c_out, f_out, r_out, a_out = fs_out.shape
        fs_out = fs_out.permute(0, 2, 1, 3, 4).reshape(-1, c_out, r_out, a_out)
        fs_out = F.interpolate(fs_out, (r, a), mode='bilinear', align_corners=True).reshape(b_out, f_out, c_out, r, a).permute(0, 2, 1, 3, 4)
        if self.training:
            b_out, c_out, f_out, r_out, a_out = f3_out.shape
            f3_out = f3_out.permute(0, 2, 1, 3, 4).reshape(-1, c_out, r_out, a_out)
            f3_out = F.interpolate(f3_out, (r, a), mode='bilinear', align_corners=True).reshape(b_out, f_out, c_out, r, a).permute(0, 2, 1, 3, 4)

            b_out, c_out, f_out, r_out, a_out = f4_out.shape
            f4_out = f4_out.permute(0, 2, 1, 3, 4).reshape(-1, c_out, r_out, a_out)
            f4_out = F.interpolate(f4_out, (r, a), mode='bilinear', align_corners=True).reshape(b_out, f_out, c_out, r, a).permute(0, 2, 1, 3, 4)
            return [f3_out, f4_out, fs_out]
        else:
            return fs_out

