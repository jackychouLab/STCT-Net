import torch.nn as nn
import torch
from .modules.mnet import MNet
from .backbones.cdc import RadarVanilla
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from .modules.micro_embedding import GAM, MicroEmbedding



class SimMNet(nn.Module):
    def __init__(self, in_chirps, out_channels, use_branch=False, stft_cfg=[128, 64], branch_cfg=['normal', 0]):
        super(SimMNet, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels

        self.s_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.SiLU(inplace=True),
        )
        self.t_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.SiLU(inplace=True),
        )
        self.m_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.SiLU(inplace=True),
        )
        self.use_branch = use_branch
        if self.use_branch:
            self.proj = nn.Sequential(
                nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(out_channels // 4, out_channels),
                nn.SiLU(inplace=True),
            )
            self.micro = MicroEmbedding(out_channels, win_len=stft_cfg[0], hop_len=stft_cfg[1], save_type=branch_cfg[0], radius=branch_cfg[1])

    def forward(self, x):
        b, c, f, cs, r, a = x.shape
        x_out = x.new_empty(b, self.out_channels, f, r, a)
        if self.use_branch:
            x_micro = x.new_empty(b, f, cs, r, a)
        for sub_f in range(f):
            x_sub = x[:, :, sub_f, :, :, :].contiguous()
            x_s = self.s_conv3d(x_sub)
            x_t = self.t_conv3d(x_sub)
            x_out[:, :, sub_f, :, :] = torch.cat([torch.mean(x_s, dim=-3, keepdim=False), torch.mean(x_t, dim=-3, keepdim=False)], 1)
            if self.use_branch:
                x_micro[:, sub_f, :, :, :] = torch.mean(self.proj(torch.cat([x_s, x_t], dim=1)), dim=1, keepdim=False)
        x_out = self.m_conv3d(x_out)
        if self.use_branch:
            return x_out, x_micro
        else:
            return x_out


class simDecoder(nn.Module):
    def __init__(self, n_class, train_type):
        super(simDecoder, self).__init__()
        self.train_type = train_type
        self.up_sample1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.GroupNorm(512 // 4, 512, 1e-6),
            nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.merge_1 = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(256 // 4, 256, 1e-6),
            nn.SiLU(inplace=True),
        )
        self.up_sample2 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.GroupNorm(256 // 4, 256, 1e-6),
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.merge_2 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(128 // 4, 128, 1e-6),
            nn.SiLU(inplace=True),
        )
        self.up_sample3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.GroupNorm(128 // 4, 128, 1e-6),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.merge_3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(64 // 4, 64, 1e-6),
            nn.SiLU(inplace=True),
        )
        self.up_sample4 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.GroupNorm(64 // 4, 64, 1e-6),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(128 // 4, 128, 1e-6),
            nn.SiLU(inplace=True),
            nn.Conv3d(128, n_class, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if self.train_type=='multi':
            self.aux_head_1 = nn.Sequential(
                nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(128 // 4, 128, 1e-6),
                nn.SiLU(inplace=True),
                nn.Conv3d(128, n_class, kernel_size=3, stride=1, padding=1, bias=True),
            )
            self.aux_head_2 = nn.Sequential(
                nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(128 // 4, 128, 1e-6),
                nn.SiLU(inplace=True),
                nn.Conv3d(128, n_class, kernel_size=3, stride=1, padding=1, bias=True),
            )
            self.aux_head_3 = nn.Sequential(
                nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(128 // 4, 128, 1e-6),
                nn.SiLU(inplace=True),
                nn.Conv3d(128, n_class, kernel_size=3, stride=1, padding=1, bias=True),
            )

    def forward(self, x):
        out = []
        out.append(self.merge_1(torch.concat([self.up_sample1(x[3]), x[2]], dim=1)))
        out.append(self.merge_2(torch.concat([self.up_sample2(out[0]), x[1]], dim=1)))
        out.append(self.merge_3(torch.concat([self.up_sample3(out[1]), x[0]], dim=1)))
        out.append(self.head(self.up_sample4(out[2])))
        if self.train_type == 'single':
            return out[-1]
        if self.train_type == 'multi':
            if self.training:
                out[0] = self.aux_head_1(out[0])
                out[1] = self.aux_head_2(out[1])
                out[2] = self.aux_head_3(out[2])
                return out
            else:
                return out[-1]


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1, 1)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, drop_out_rate=0.15):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop_out_rate)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2, **kwargs):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv3d(in_channels=dim, out_channels=med_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.dwconv = nn.Conv3d(in_channels=med_channels, out_channels=med_channels, kernel_size=(5, 7, 7), stride=(1, 1, 1), padding=(2, 3, 3), bias=True, groups=med_channels)
        self.pwconv2 = nn.Conv3d(in_channels=med_channels, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, num_heads=1, attn_drop_out_rate=0., proj_drop_rate=0., **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, True)
        self.attn_drop = nn.Dropout(attn_drop_out_rate)
        self.proj = nn.Linear(self.attention_dim, dim, True)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, C, F, R, A = x.shape
        N = F * R * A
        x = x.permute(0, 2, 3, 4, 1)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, F, R, A, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class MetaFormerBlock(nn.Module):
    def __init__(self, dim, token_mixer=nn.Identity, mlp=Mlp, drop_path=0., res_scale_init_value=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(dim // 4, dim, eps=1e-6)
        self.token_mixer = token_mixer(dim=dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
        self.norm2 = nn.GroupNorm(dim // 4, dim, eps=1e-6)
        self.mlp = mlp(dim=dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.drop_path1(self.token_mixer(self.norm1(x)))
        x = self.res_scale2(x) + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class CAFormer(nn.Module):
    def __init__(self, in_chans, depths=[3, 3, 9, 3], dims=[64, 128, 256, 512], use_branch=False, stft_cfg=[128, 64]):
        super().__init__()
        num_stage = len(depths)
        self.num_stage = num_stage
        token_mixers = [SepConv, SepConv, Attention, Attention]
        drop_path_rate = 0.
        res_scale_init_values = [None, None, 1.0, 1.0]
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv3d(in_channels=in_chans, out_channels=dims[0], kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False),
                nn.GroupNorm(dims[0] // 4, dims[0], eps=1e-6),
            )
        )
        self.downsample_layers.append(
            nn.Sequential(
                nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.GroupNorm(dims[0] // 4, dims[0], eps=1e-6),
                nn.Conv3d(in_channels=dims[0], out_channels=dims[1], kernel_size=1, stride=1, padding=0, bias=True),
                nn.SiLU(inplace=True),
            )
        )
        self.downsample_layers.append(
            nn.Sequential(
                nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.GroupNorm(dims[1] // 4, dims[1], eps=1e-6),
                nn.Conv3d(in_channels=dims[1], out_channels=dims[2], kernel_size=1, stride=1, padding=0, bias=True),
                nn.SiLU(inplace=True),
            )
        )
        self.downsample_layers.append(
            nn.Sequential(
                nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.GroupNorm(dims[2] // 4, dims[2], eps=1e-6),
                nn.Conv3d(in_channels=dims[2], out_channels=dims[3], kernel_size=1, stride=1, padding=0, bias=True),
                nn.SiLU(inplace=True),
            )
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                                  token_mixer=token_mixers[i],
                                  mlp=Mlp,
                                  drop_path=dp_rates[cur + j],
                                  res_scale_init_value=res_scale_init_values[i],
                                  ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.use_branch = use_branch
        if self.use_branch:
            self.gam_list = nn.ModuleList()
            self.gam_list.append(GAM(64, stft_cfg[0] // 2 + 1))
            self.gam_list.append(GAM(128, stft_cfg[0] // 2 + 1))
            self.gam_list.append(GAM(256, stft_cfg[0] // 2 + 1))
            self.gam_list.append(GAM(512, stft_cfg[0] // 2 + 1))

    def forward(self, x):
        if self.use_branch:
            x, x_micro = x
            out = []
            for i in range(self.num_stage):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                x = self.gam_list[i](x, x_micro[i])
                out.append(x)
        else:
            out = []
            for i in range(self.num_stage):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
                out.append(x)
        return out


class RODNetCDC_refine(nn.Module):
    def __init__(self, in_channels, n_class, train_type, mnet_type='sim', backbone_type='cdc', mnet_cfg=None, use_branch=False, branch_cfg=['normal', 0], stft_cfg=[128, 64]):
        super(RODNetCDC_refine, self).__init__()
        in_chirps_mnet, out_channels_mnet = mnet_cfg
        self.backbone_type = backbone_type
        assert in_channels == in_chirps_mnet
        if mnet_type == 'sim':
            self.mnet = SimMNet(in_chirps_mnet, out_channels_mnet, use_branch=use_branch, branch_cfg=branch_cfg, stft_cfg=stft_cfg)
        elif mnet_type == 'mnet':
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet)
        if backbone_type == 'cdc':
            self.prd = RadarVanilla(out_channels_mnet, n_class, use_mse_loss=True)
        if backbone_type == 'ca':
            self.prd = CAFormer(out_channels_mnet, use_branch=use_branch, stft_cfg=stft_cfg)
            self.dec = simDecoder(n_class, train_type)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mnet(x)
        if self.backbone_type == 'cdc':
            x = self.prd(x)
        if self.backbone_type == 'ca':
            x = self.prd(x)
            x = self.dec(x)
        return x