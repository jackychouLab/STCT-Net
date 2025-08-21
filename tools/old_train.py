import torch
import torch.nn.functional as F
import torch.nn as nn

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

# 初始化
B, C, T, H, W = 1, 256, 8, 64, 64
x = torch.randn(B, C, T, H, W, dtype=torch.float32)

# 三线性原生
y_native = F.interpolate(x, scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

# 模拟版
model = myTU(time_scale=2, space_scale=(2, 2), align_corners=False)
y_mine = model(x)

# 误差比较
print("Abs diff:", torch.abs(y_native - y_mine).max())
print("Allclose:", torch.allclose(y_native, y_mine, atol=1e-7))