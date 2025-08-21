import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
import math
import torch.nn.functional as F



class Scale(nn.Module):

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class StarReLU(nn.Module):

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class LayerNorm(nn.Module):

    def __init__(self, num_groups, dims, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dims, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


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


class myTU(nn.Module):
    def __init__(self, time_scale=2, space_scale=(2, 2), align_corners=False):
        super().__init__()
        self.time_scale = (time_scale, 1)
        self.space_scale = space_scale
        self.align_corners = align_corners

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(B * H * W, C, T).unsqueeze(-1)
        x = F.interpolate(x, scale_factor=self.time_scale, mode='bilinear', align_corners=self.align_corners).squeeze(-1)
        _, _, T_new = x.shape
        x = x.reshape(B, H, W, C, T_new).permute(0, 4, 3, 1, 2).reshape(B * T_new, C, H, W)
        x = F.interpolate(x, scale_factor=self.space_scale, mode='bilinear', align_corners=self.align_corners)
        _, _, H_new, W_new = x.shape
        x = x.reshape(B, T_new, C, H_new, W_new).permute(0, 2, 1, 3, 4)

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

    def __init__(self, n_class, train_type, head_size, norm, act, full_conv, depths=[2, 2]):
        super(RODDecode, self).__init__()

        assert train_type in ['single', 'multi']
        self.train_type = train_type

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
            nn.Sigmoid()
        )
        if self.train_type == 'multi':
            self.sub_head_1 = nn.Sequential(
                norm(128 // 4, 128, 1e-6),
                nn.Conv3d(in_channels=128, out_channels=n_class, kernel_size=head_size, stride=1, padding=((head_size[0] - 1) // 2, (head_size[1] - 1) // 2, (head_size[2] - 1) // 2), bias=True),
                nn.Sigmoid()
            )
            self.sub_head_2 = nn.Sequential(
                norm(16, 64, 1e-6),
                nn.Conv3d(in_channels=64, out_channels=n_class, kernel_size=head_size, stride=1, padding=((head_size[0] - 1) // 2, (head_size[1] - 1) // 2, (head_size[2] - 1) // 2), bias=True),
                nn.Sigmoid()
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

        if self.train_type == 'single':
            return out3
        if self.train_type == 'multi':
            if self.training:
                out1 = self.sub_head_1(out1)
                out2 = self.sub_head_2(out2)
                return [out1, out2, out3]
            else:
                return out3


class MNetv1(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv1, self).__init__()
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


class MNetv2(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv2, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.v_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(in_chirps, 1, 1), stride=(32, 1, 1), padding=0, bias=True),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU()
        )
        self.s_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(in_chirps, 1, 1))
        )
        self.merge_net = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int(out_channels)),
            nn.GELU()
        )

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_velocity = self.v_net(x[:, :, win, :, :, :]).squeeze(2)
            x_space = self.s_net(x[:, :, win, :, :, :]).squeeze(2)
            x_merge = torch.cat([x_velocity, x_space], dim=1)
            x_out[:, :, win, :, :] = self.merge_net(x_merge)
        return x_out


class MNetv3(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv3, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.conv_op = nn.Conv3d
        self.t_conv3d = nn.Conv3d(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=True)
        t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        self.t_avgpool = nn.AvgPool3d(kernel_size=(t_conv_out, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = self.t_conv3d(x[:, :, win, :, :, :])
            x_win = self.t_avgpool(x_win)
            x_win = x_win.view(batch_size, self.out_channels, w, h)
            x_out[:, :, win, ] = x_win
        return x_out


class MNetv4(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv4, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.conv_op = nn.Conv3d
        self.t_conv3d = nn.Conv3d(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=True)
        t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        self.t_avgpool = nn.AvgPool3d(kernel_size=(t_conv_out, 1, 1))
        self.t_maxpool = nn.MaxPool3d(kernel_size=(t_conv_out, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = self.t_conv3d(x[:, :, win, :, :, :])
            x_win = self.t_maxpool(x_win) + self.t_avgpool(x_win)
            x_win = x_win.view(batch_size, self.out_channels, w, h)
            x_out[:, :, win, ] = x_win
        return x_out


class MNetv5(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv5, self).__init__()
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


class MNetv6(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv6, self).__init__()
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
            nn.MaxPool3d((in_chirps, 1, 1))
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


class MNetv7(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv7, self).__init__()
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
        )

        self.m_net = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=True),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.GELU()
        )

        self.max = nn.MaxPool3d((in_chirps, 1, 1))
        self.avg = nn.AvgPool3d((in_chirps, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = x[:, :, win, :, :, :]
            x_c = x_win[:, 0, ...] + 1j * x_win[0, 1, ...]
            x_mag = torch.abs(torch.fft.fft(x_c, dim=1))
            x_d = self.d_net(x_mag)
            x_s = self.s_net(x_win)
            x_s = (self.avg(x_s) + self.max(x_s)).squeeze(2)
            x_out[:, :, win, :, :] = self.m_net(torch.cat([x_d, x_s], dim=1))
        return x_out


class MNetv8(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv8, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.v_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(in_chirps, 1, 1), stride=(32, 1, 1), padding=0, bias=True),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU()
        )
        self.s_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU(),
            nn.AvgPool3d(kernel_size=(in_chirps, 1, 1))
        )
        self.merge_net = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int(out_channels)),
            nn.GELU()
        )

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_velocity = self.v_net(x[:, :, win, :, :, :]).squeeze(2)
            x_space = self.s_net(x[:, :, win, :, :, :]).squeeze(2)
            x_merge = torch.cat([x_velocity, x_space], dim=1)
            x_out[:, :, win, :, :] = self.merge_net(x_merge)
        return x_out


class MNetv9(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv9, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.v_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(in_chirps, 1, 1), stride=(32, 1, 1), padding=0, bias=True),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU()
        )
        self.s_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU(),
        )
        self.merge_net = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int(out_channels)),
            nn.GELU()
        )

        self.max_pool = nn.MaxPool3d(kernel_size=(in_chirps, 1, 1))
        self.avg_pool = nn.AvgPool3d(kernel_size=(in_chirps, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_velocity = self.v_net(x[:, :, win, :, :, :]).squeeze(2)  # (B, C/2, 128, 128)
            x_space = self.s_net(x[:, :, win, :, :, :])  # (B, C/2, 128, 128)
            x_space = (self.avg_pool(x_space) + self.max_pool(x_space)).squeeze(2)
            x_merge = torch.cat([x_velocity, x_space], dim=1)  # (B, C, 128, 128)
            x_out[:, :, win, :, :] = self.merge_net(x_merge)
        return x_out


class MNetv10(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv10, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.s_net = nn.Sequential(
            nn.Conv3d(2, out_channels, (1, 3, 3), 1, (0, 1, 1), bias=True),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.GELU(),
            nn.AvgPool3d((in_chirps, 1, 1))
        )


    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = x[:, :, win, :, :, :]
            x_s = self.s_net(x_win).squeeze(2)
            x_out[:, :, win, :, :] = x_s
        return x_out


class MNetv11(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv11, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.d_net = nn.Sequential(
            nn.Conv2d(in_chirps, out_channels, 1, 1, 0, bias=True),
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
            x_out[:, :, win, :, :] = x_d
        return x_out


class MNetv12(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(MNetv12, self).__init__()
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


    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_win = x[:, :, win, :, :, :]
            x_c = x_win[:, 0, ...] + 1j * x_win[0, 1, ...]
            x_mag = torch.abs(torch.fft.fft(x_c, dim=1))
            x_d = self.d_net(x_mag)
            x_s = self.s_net(x_win).squeeze(2)
            x_out[:, :, win, :, :] = torch.cat([x_d, x_s], dim=1)
        return x_out


# 1-MNet(max) 2-E-RODNet(max) 3-MNet(avg) 4-MNet(mix) 5-STCT(avg) 6-STCT(max) 7-STCT(mix) 8-E-RODNet(avg) 9-E-RODNet(mix) 10-S 11-T 12-ST
MNet = {'1':MNetv1, '2':MNetv2, '3':MNetv3, '4':MNetv4, '5':MNetv5, '6':MNetv6, '7':MNetv7, '8':MNetv8, '9':MNetv9, '10':MNetv10, '11':MNetv11, '12':MNetv12}

class myNet(nn.Module):
    def __init__(self, mnet_cfg, n_class, mnet_type, train_type, head_size, norm_type, act_type, full_conv):
        super(myNet, self).__init__()
        assert norm_type in ['gn', 'ln']
        if norm_type == 'gn':
            norm = nn.GroupNorm
        if norm_type == 'ln':
            norm = LayerNorm
        assert act_type in ['starrelu', 'gelu']
        if act_type == 'gelu':
            act = nn.GELU
        if act_type == 'starrelu':
            act = StarReLU
        self.mnet = MNet[f'{mnet_type}'](in_chirps=mnet_cfg[0], out_channels=mnet_cfg[1])
        self.encoder = RODEncode(mnet_cfg[1], norm, act, full_conv)
        self.decoder = RODDecode(n_class, train_type, head_size, norm, act, full_conv)

    def forward(self, x):
        x = self.mnet(x)
        x, x1, x2 = self.encoder(x)
        x = self.decoder(x, x1, x2)
        return x