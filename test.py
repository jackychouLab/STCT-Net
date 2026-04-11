import torch
import torch.nn as nn



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

class TrilinearUpsample(nn.Module):
    def __init__(
        self,
        scale_factor: Union[int, Tuple[int, int, int]],
        align_corners: bool = False,
    ):
        super().__init__()
        # 处理 scale_factor
        if isinstance(scale_factor, int):
            self.sf_d = self.sf_h = self.sf_w = scale_factor
        else:
            assert len(scale_factor) == 3 and all(isinstance(s, int) and s > 0 for s in scale_factor), \
                "`scale_factor` 应为 3 个正整数的元组"
            self.sf_d, self.sf_h, self.sf_w = scale_factor

        self.align_corners = align_corners

    def _compute_mapping(self, L_in: int, L_out: int, sf: int):
        if self.align_corners and L_out > 1:
            scale = (L_in - 1) / (L_out - 1)
            coords = torch.arange(L_out, device=self._device) * scale
        else:
            coords = (torch.arange(L_out, device=self._device) + 0.5) / sf - 0.5

        idx0 = torch.clamp(torch.floor(coords).long(), 0, L_in - 1)
        idx1 = torch.clamp(idx0 + 1, 0, L_in - 1)
        w = (coords - idx0.float())
        return idx0, idx1, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, "输入必须是 5D 张量 (N, C, D, H, W)"
        N, C, D, H, W = x.shape
        self._device = x.device  # 给 mapping 用


        D_out, H_out, W_out = D * self.sf_d, H * self.sf_h, W * self.sf_w
        w0, w1, tw = self._compute_mapping(W, W_out, self.sf_w)
        tw = tw.view(1, 1, 1, 1, W_out)
        w0_idx = w0.view(1,1,1,1,W_out).expand(N, C, D, H, W_out)
        w1_idx = w1.view(1,1,1,1,W_out).expand(N, C, D, H, W_out)
        v0 = torch.gather(x, dim=4, index=w0_idx)
        v1 = torch.gather(x, dim=4, index=w1_idx)
        inter_w = v0 * (1 - tw) + v1 * tw  # (N,C,D,H,W_out)

        h0, h1, th = self._compute_mapping(H, H_out, self.sf_h)
        th = th.view(1, 1, 1, H_out, 1)
        h0_idx = h0.view(1,1,1,H_out,1).expand(N, C, D, H_out, W_out)
        h1_idx = h1.view(1,1,1,H_out,1).expand(N, C, D, H_out, W_out)
        u0 = torch.gather(inter_w, dim=3, index=h0_idx)
        u1 = torch.gather(inter_w, dim=3, index=h1_idx)
        inter_h = u0 * (1 - th) + u1 * th  # (N,C,D,H_out,W_out)

        z0, z1, tz = self._compute_mapping(D, D_out, self.sf_d)
        tz = tz.view(1, 1, D_out, 1, 1)
        z0_idx = z0.view(1,1,D_out,1,1).expand(N, C, D_out, H_out, W_out)
        z1_idx = z1.view(1,1,D_out,1,1).expand(N, C, D_out, H_out, W_out)
        w0_ = torch.gather(inter_h, dim=2, index=z0_idx)
        w1_ = torch.gather(inter_h, dim=2, index=z1_idx)
        out = w0_ * (1 - tz) + w1_ * tz  # (N,C,D_out,H_out,W_out)


        return out


# ========== 验证一致性 ==========
torch.manual_seed(0)

x = torch.randn(1, 3, 5, 6, 7, requires_grad=True)  # [N, C, D, H, W]

# 原生 PyTorch 三线性上采样
y_ref = F.interpolate(x, scale_factor=(2, 2, 2), mode="trilinear", align_corners=False)

# 手写三线性上采样
t1 = TrilinearUpsample((2, 2, 2), align_corners=False)
y_test = t1(x)

# 误差
diff = (y_ref - y_test).abs().max()

print("原生 F.interpolate 输出大小:", y_ref.shape)
print("手写 trilinear_upsample 输出大小:", y_test.shape)
print("两者最大误差:", diff.item())