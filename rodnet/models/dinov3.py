import torch
import torch.nn as nn
import timm
import torch.nn.functional as F



class SimMNet(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(SimMNet, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels

        self.s_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.SiLU(inplace=True),
            nn.AvgPool3d(kernel_size=(in_chirps, 1, 1)),
        )
        self.t_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.SiLU(inplace=True),
            nn.AvgPool3d(kernel_size=(in_chirps, 1, 1)),
        )

        self.m_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(3),
            nn.SiLU(inplace=True),
        )


    def forward(self, x):
        b, c, win, chirps, h, w = x.shape
        x_m = torch.zeros((b, self.out_channels, win, w, h)).cuda()
        for win in range(win):
            x_win_t = self.t_conv3d(x[:, :, win, :, :, :]).squeeze(2)
            x_win_s = self.s_conv3d(x[:, :, win, :, :, :]).squeeze(2)
            x_m[:, :, win, :, :] = torch.cat([x_win_t, x_win_s], dim=1)
        out = self.m_conv3d(x_m)
        return out



class simDINOv3(nn.Module):
    def __init__(self, in_channels, n_class):
        super(simDINOv3, self).__init__()
        self.mnet = SimMNet(32, 32)
        self.backbone = timm.create_model('convnext_base.dinov3_lvd1689m', pretrained=True ,features_only=True, cache_dir="/home/jackychou/code/cache/")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.fpn_in_4 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
        )
        self.fpn_in_3 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
        )
        self.fpn_in_2 = nn.Sequential(
            nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
        )
        self.fpn_in_1 = nn.Sequential(
            nn.Conv3d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
        )

        self.fpn_mix_1 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
        )
        self.fpn_mix_2 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
        )
        self.fpn_mix_3 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
        )
        self.fpn_mix_4 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
        )
        self.fpn_mix_5 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=n_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.mnet(x)
        b, c, f, r, a = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, r, a)

        self.backbone.eval()
        x = self.backbone(x)
        x4, x3, x2, x1 = x
        del x
        x1 = self.fpn_in_1(x1.view(b, f, 1024, 4, 4).transpose(1, 2))
        x2 = self.fpn_in_2(x2.view(b, f, 512, 8, 8).transpose(1, 2))
        x3 = self.fpn_in_3(x3.view(b, f, 256, 16, 16).transpose(1, 2))
        x4 = self.fpn_in_4(x4.view(b, f, 128, 32, 32).transpose(1, 2))
        out = self.fpn_mix_1(F.interpolate(x1, scale_factor=(1, 2, 2), mode='nearest') + x2)
        out = self.fpn_mix_2(F.interpolate(out, scale_factor=(1, 2, 2), mode='nearest') + x3)
        out = self.fpn_mix_3(F.interpolate(out, scale_factor=(1, 2, 2), mode='nearest') + x4)
        out = self.fpn_mix_4(F.interpolate(out, scale_factor=(1, 2, 2), mode='nearest'))
        out = self.fpn_mix_5(F.interpolate(out, scale_factor=(1, 2, 2), mode='nearest'))

        return out

