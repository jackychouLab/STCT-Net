from torch import nn
import torch
from torch.nn import functional as F



class MicroEmbedding(nn.Module):
    def __init__(self, in_channels, win_len, hop_len, save_type='normal', radius=0, out_put_list=[[4, 8, 8]]):
        super().__init__()
        assert save_type in ['normal', 'high', 'low']
        if save_type in ['high', 'low']:
            assert radius in [1, 2, 3, 4, 5]
        self.proj_list = nn.ModuleList()
        for _ in range(len(out_put_list)):
            self.proj_list.append(
                nn.Sequential(
                    nn.GroupNorm(in_channels // 4, in_channels, eps=1e-6),
                    nn.Conv3d(in_channels, 1, 1, 1, 0, bias=True),
                    nn.SiLU(inplace=True),
                )
            )
        self.win_len = win_len
        self.hop_len = hop_len
        self.register_buffer("window", torch.hann_window(win_len), persistent=False)
        self.out_put_list = out_put_list
        self.type = save_type
        self.radius = radius

    def forward(self, x):
        b, f, c, chirps, r, a = x.shape
        x = x.reshape(b * f, c, chirps, r, a)
        window = self.window.to(device=x.device, dtype=x.dtype)
        out = []
        for i in range(len(self.out_put_list)):
            mag = F.adaptive_avg_pool3d(x, output_size=[chirps] + self.out_put_list[i][1:])
            mag = self.proj_list[i](mag).squeeze(1).reshape(b, f * chirps, self.out_put_list[i][1] * self.out_put_list[i][2]).transpose(1, 2).reshape(-1, f * chirps).contiguous()
            stft_out = torch.stft(mag, n_fft=self.win_len, hop_length=self.hop_len, window=window, return_complex=False, center=False, onesided=True)
            mag = torch.sqrt(stft_out[..., 0] ** 2 + stft_out[..., 1] ** 2 + 1e-6)
            freq_bins, times_steps = mag.shape[1:]
            mag = mag.reshape(b, self.out_put_list[i][1], self.out_put_list[i][2], freq_bins, times_steps).permute(0, 3, 4, 1, 2).contiguous()
            mag = F.adaptive_avg_pool3d(mag, output_size=self.out_put_list[i])
            out.append(mag)
        if self.type != "normal":
            for i in range(len(out)):
                r, a = out[i].shape[-2:]
                cr, ca = r // 2, a // 2
                r0, r1 = cr - self.radius, cr + self.radius
                a0, a1 = ca - self.radius, ca + self.radius
                r0, r1 = max(0, r0), min(r, r1)
                a0, a1 = max(0, a0), min(a, a1)
                if self.type == "high":
                    mask = torch.ones((r, a), device=out[i].device, dtype=out[i].dtype)
                    mask[r0:r1, a0:a1] = 0.
                if self.type == "low":
                    mask = torch.zeros((r, a), device=out[i].device, dtype=out[i].dtype)
                    mask[r0:r1, a0:a1] = 1.
                out[i] = torch.fft.fftn(out[i], dim=(-2, -1))
                out[i] = torch.fft.fftshift(out[i], dim=(-2, -1))
                out[i] = out[i] * mask[None, None, None, :, :]
                out[i] = torch.fft.ifftn(torch.fft.ifftshift(out[i], dim=(-2, -1)), dim=(-2, -1)).real
        return out


class GAM(nn.Module):
    def __init__(self, encoder_channles, micro_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(micro_channels, encoder_channles, 1, 1, 0, bias=False),
            nn.GroupNorm(encoder_channles // 4, encoder_channles, eps=1e-6),
            nn.SiLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Conv3d(encoder_channles * 2, encoder_channles // 2, 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(encoder_channles // 2, encoder_channles, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(encoder_channles, encoder_channles, 3, 1, 1, bias=False),
            nn.GroupNorm(encoder_channles // 4, encoder_channles, eps=1e-6),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, micro_x):
        micro_x = self.proj(micro_x)
        fusion_input = torch.cat([x, micro_x], dim=1)
        gate = self.gate(fusion_input)
        fused = x + gate * micro_x
        out = self.fusion(fused)
        return out