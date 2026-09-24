from functools import reduce, lru_cache
from operator import mul
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, T, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size*window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, T, H, W): #(B*num_windows, window_size, window_size, window_size, C)-->(B, T, H, W, C)
    """
    Args:
        windows: (B*num_windows, window_size, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, T, H, W, C)
    """
    x = windows.view(B, T // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift=False, shift_type='psm'):

        super().__init__()
        self.dim = dim
        ## for bayershift
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.shift = shift
        self.shift_type = shift_type

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        # Do the same rotation to coords
        coords_old = coords.clone()

        ## pattern patternC - 9
        coords[:, :, 0::3, 0::3] = torch.roll(coords[:, :, 0::3, 0::3], shifts=-4, dims=1)
        coords[:, :, 0::3, 1::3] = torch.roll(coords[:, :, 0::3, 1::3], shifts=1, dims=1)
        coords[:, :, 0::3, 2::3] = torch.roll(coords[:, :, 0::3, 2::3], shifts=2, dims=1)
        coords[:, :, 1::3, 2::3] = torch.roll(coords[:, :, 1::3, 2::3], shifts=3, dims=1)
        coords[:, :, 1::3, 0::3] = torch.roll(coords[:, :, 1::3, 0::3], shifts=-1, dims=1)
        coords[:, :, 2::3, 0::3] = torch.roll(coords[:, :, 2::3, 0::3], shifts=-2, dims=1)
        coords[:, :, 2::3, 1::3] = torch.roll(coords[:, :, 2::3, 1::3], shifts=-3, dims=1)
        coords[:, :, 2::3, 2::3] = torch.roll(coords[:, :, 2::3, 2::3], shifts=4, dims=1)

        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        coords_old_flatten = torch.flatten(coords_old, 1) # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords_old = coords_old_flatten[:, :, None] - coords_old_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords_old = relative_coords_old.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3


        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords_old[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords_old[:, :, 1] += self.window_size[1] - 1
        relative_coords_old[:, :, 2] += self.window_size[2] - 1


        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        relative_coords_old[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords_old[:, :, 1] *= (2 * self.window_size[2] - 1)


        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        relative_position_index_old = relative_coords_old.sum(-1)   # Wd*Wh*Ww, Wd*Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)
        self.register_buffer("relative_position_index_old", relative_position_index_old)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.ratio = nn.Parameter(torch.tensor(0.5))

        if self.shift and self.shift_type == 'psm':
            self.shift_op = PatchShift(self.num_heads, False, 1)     #得到shift_op操作  输入维度(B_,nH,N,C//nH) 输出维度(B_,N,C)
            self.shift_op_back = PatchShift(self.num_heads, True, 1)
        elif self.shift and self.shift_type == 'tsm':
            self.shift_op = TemporalShift(8)      #得到shift_op操作  输入维度(B_,nH,N,C//nH) 输出维度(B_,N,C)


    def forward(self, x, mask=None, batch_size=8, frame_len=8):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape

        if self.shift:
            x = x.view(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            x = self.shift_op(x, batch_size, frame_len)
            x = x.permute(0, 2, 1, 3).reshape(B_, N, C)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)    # B_, nH, N, N  (B_=B*nW ,N=Wd*Wh*Ww)

        if self.shift and self.shift_type == 'psm':
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(
                N, N, -1)   # Wd*Wh*Ww,Wd*Wh*Ww,nH
        else:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index_old[:N, :N].reshape(-1)].reshape(
                N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N


        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.shift and self.shift_type == 'psm':
            x = self.shift_op_back(attn @ v, batch_size, frame_len).transpose(1, 2).reshape(B_, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        return x, v, k, q



class PatchShift(nn.Module):
    def __init__(self, n_div=8, inv=False, ratio=1):
        super(PatchShift, self).__init__()
        self.fold_div = n_div
        self.inv = inv
        self.ratio = ratio

    def forward(self, x, batch_size, frame_len):
        x = self.shift(x, fold_div=self.fold_div, inv=self.inv, ratio=self.ratio, batch_size=batch_size,
                       frame_len=frame_len)
        return x  # self.net(x)

    @staticmethod
    def shift(x, fold_div=3, inv=False, ratio=0.5, batch_size=8, frame_len=8):
        B, num_heads, N, c = x.size()
        fold = int(num_heads * ratio)
        feat = x
        feat = feat.view(batch_size, frame_len, -1, num_heads, 8, 8, c)
        out = feat.clone()
        multiplier = 1
        stride = 1
        if inv:
            multiplier = -1

        ## Pattern C
        out[:, :, :, :fold, 0::3, 0::3, :] = torch.roll(feat[:, :, :, :fold, 0::3, 0::3, :],
                                                        shifts=-4 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 0::3, 1::3, :] = torch.roll(feat[:, :, :, :fold, 0::3, 1::3, :], shifts=multiplier * stride,
                                                        dims=1)
        out[:, :, :, :fold, 1::3, 0::3, :] = torch.roll(feat[:, :, :, :fold, 1::3, 0::3, :],
                                                        shifts=-multiplier * stride, dims=1)
        out[:, :, :, :fold, 0::3, 2::3, :] = torch.roll(feat[:, :, :, :fold, 0::3, 2::3, :],
                                                        shifts=2 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 2::3, 0::3, :] = torch.roll(feat[:, :, :, :fold, 2::3, 0::3, :],
                                                        shifts=-2 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 1::3, 2::3, :] = torch.roll(feat[:, :, :, :fold, 1::3, 2::3, :],
                                                        shifts=3 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 2::3, 1::3, :] = torch.roll(feat[:, :, :, :fold, 2::3, 1::3, :],
                                                        shifts=-3 * multiplier * stride, dims=1)
        out[:, :, :, :fold, 2::3, 2::3, :] = torch.roll(feat[:, :, :, :fold, 2::3, 2::3, :],
                                                        shifts=4 * multiplier * stride, dims=1)

        out = out.view(B, num_heads, N, c)
        return out


class TemporalShift(nn.Module):
    def __init__(self, n_div=8):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div

    def forward(self, x, batch_size, frame_len):
        x = self.shift(x, fold_div=self.fold_div, batch_size=batch_size, frame_len=frame_len)
        return x

    @staticmethod
    def shift(x, fold_div=8, batch_size=8, frame_len=8):
        B, num_heads, N, c = x.size()
        fold = c // fold_div
        feat = x
        feat = feat.view(batch_size, frame_len, -1, num_heads, N, c)
        out = feat.clone()

        out[:, 1:, :, :, :, :fold] = feat[:, :-1, :, :, :, :fold]  # shift left
        out[:, :-1, :, :, :, fold:2 * fold] = feat[:, 1:, :, :, :, fold:2 * fold]  # shift right

        out = out.view(B, num_heads, N, c)

        return out


class SemanticAttention(nn.Module):
    """ ClassMasking
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, num_cls):

        super().__init__()
        self.dim = dim
        self.num_cls = num_cls
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_cls_q = nn.Linear(self.dim, self.num_cls)
        self.mlp_cls_k = nn.Linear(self.dim, self.num_cls)

        self.mlp_v = nn.Linear(self.dim, self.dim)

        self.mlp_res = nn.Linear(self.dim, self.dim)

        self.proj_drop = nn.Dropout(0.1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.init_weight()

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, N, C)
        returns:
            class_seg_map: (B, N, K)
            gated feats: (B, N, C)
        """

        B, N, C = x.shape
        seg_map = self.mlp_cls_q(x) #(B,N,K)
        seg_ft = self.mlp_cls_k(x) #(B,N,K)

        feats = self.mlp_v(x) #(B,N,C)

        seg_score = seg_map @ seg_ft.transpose(-2, -1)
        seg_score = self.softmax(seg_score)

        feats = seg_score @ feats   #(B,N,C)
        feats = self.mlp_res(feats)
        feats = self.proj_drop(feats)

        feat_map = self.gamma * feats + x  #(B,N,C)

        return seg_map, feat_map

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Linear):
                nn.init.kaiming_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
            elif isinstance(ly, nn.LayerNorm):
                nn.init.constant_(ly.bias, 0)
                nn.init.constant_(ly.weight, 1.0)

        nn.init.zeros_(self.mlp_res.weight)
        if not self.mlp_res.bias is None: nn.init.constant_(self.mlp_res.bias, 0)


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(4, 4, 4), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False, shift=False, shift_type='psm'):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.shift = shift
        self.shift_type = shift_type

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, shift=self.shift,
            shift_type=self.shift_type)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward_part1(self, x, mask_matrix):
        B, L, C = x.shape
        T, H, W = self.T, self.H, self.W
        assert L == T * H * W, "input feature has wrong size"
        window_size, shift_size = get_window_size((T, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        x = x.view(B, T, H, W, C)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - T % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Tp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows,  v, k, q = self.attn(x_windows, mask=attn_mask, batch_size=B, frame_len=T)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Tp, Hp, Wp)  # B T' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :T, :H, :W, :].contiguous()

        x = x.view(B, T * H * W, C)

        return x, v, k, q


    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        v, k, q = None, None, None

        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x, v, k, q = self.forward_part1(x, mask_matrix)

        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x, v, k, q


class SWSemanticMaskBlock3D(nn.Module):
    def __init__(self, dim, num_cls, window_size=(4, 4, 4), num_blocks = 1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_cls = num_cls
        self.norm = norm_layer(dim)
        self.class_injection = nn.ModuleList([])

        for i in range(num_blocks):
            self.class_injection.append(SemanticAttention(dim=dim, num_cls=num_cls))

        self.T = None
        self.H = None
        self.W = None

    def forward(self, x):
        B, N, C = x.shape
        T, H, W = self.T, self.H, self.W
        assert N == T * H * W, "input feature has wrong size"
        K = self.num_cls

        x = self.norm(x)
        x = x.view(B, T, H, W, C)
        # pad feature maps to multiples of window size 此处需要根据窗的大小对特征图进行pad操作，
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Tp, Hp, Wp, _ = x.shape

        shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # B*nW, Wd*Wh*Ww, C

        # W-MSA/SW-MSA
        for blk in self.class_injection:
            sem_windows, x_windows = blk(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        x_windows = x_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(x_windows, self.window_size, B, Tp, Hp, Wp)  # B T' H' W' C

        # merge windows
        sem_windows = sem_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], K)
        shifted_sem = window_reverse(sem_windows, self.window_size, B, Tp, Hp, Wp)  # B T' H' W' K

        x = shifted_x
        sem_map = shifted_sem
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :T, :H, :W, :].contiguous()
            sem_map = sem_map[:, :T, :H, :W, :].contiguous()

        x = x.view(B, T*H*W, C)
        sem_map = sem_map.view(B, T*H*W, K)

        return sem_map, x


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=(16, 128, 128), patch_size=(2, 2, 2), in_chans=2, embed_dim=64, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.patches_resolution = patches_resolution
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.relu = nn.ReLU()

        self.conv1a = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))

        self.gn1a = nn.GroupNorm(num_groups=int(embed_dim/4), num_channels=embed_dim)
        self.gn1b = nn.GroupNorm(num_groups=int(embed_dim/4), num_channels=embed_dim)

    def forward(self, x):

        x = self.relu(self.gn1a(self.conv1a(x)))
        x = self.relu(self.gn1b(self.conv1b(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, last = False):
        super().__init__()
        self.last = last

        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        self.conv = nn.Conv3d(in_channels=dim, out_channels=dim*2,kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(num_features=dim*2)
        self.relu = nn.ReLU()

        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.gn2a  = nn.GroupNorm(num_groups=32,num_channels=128)
        self.gn2b = nn.GroupNorm(num_groups=32, num_channels=128)


        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.gn3a = nn.GroupNorm(num_groups=32, num_channels=256)
        self.gn3b = nn.GroupNorm(num_groups=32, num_channels=256)


        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(9, 5, 5), stride=(1, 2, 2), padding=(4, 2, 2))

    def forward(self, x, T, H, W):
        B, L, C = x.shape
        assert L == T * H * W, "input feature has wrong size"
        x = x.view(B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3)

        if self.last:
            x = self.relu(self.gn3a(self.conv3a(x)))
            x = self.relu(self.gn3b(self.conv3b(x)))
        else:
            x = self.relu(self.gn2a(self.conv2a(x)))
            x = self.relu(self.gn2b(self.conv2b(x)))

        x = x.permute(0, 2, 3, 4, 1)
        B1, C1 = x.shape[0], x.shape[4]
        x = x.reshape(B1, -1, C1)

        return x


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    def __init__(self,
                 i,
                 dim,
                 depth,
                 num_heads,
                 window_size=(4, 4, 4),
                 sem_window_size=(4, 4, 4),
                 num_sem_blocks=1,
                 num_cls=3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 shift_type='psm'):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_sem_blocks = num_sem_blocks
        self.shift_type = shift_type
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(dim)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                shift=True,
                shift_type='tsm' if (i % 2 == 0 and self.shift_type == 'psm') or self.shift_type == 'tsm' else 'psm',
            )
            for i in range(depth)])

        if num_sem_blocks > 0:
            self.semantic_layer = SWSemanticMaskBlock3D(dim=dim,
                                                      num_cls=num_cls,
                                                      num_blocks=num_sem_blocks,
                                                      window_size=sem_window_size)

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

        self.downsamplelast = PatchMerging(dim=dim, norm_layer=norm_layer, last=True)



    def forward(self, x, T, H, W):
        B, L, C = x.shape
        assert L == T * H * W, "input feature has wrong size"

        window_size, shift_size = get_window_size((T, H, W), self.window_size, self.shift_size)

        Tp = int(np.ceil(T / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Tp, Hp, Wp, window_size, shift_size, x.device)

        v1, k1, q1, v2, k2, q2 = None, None, None, None, None, None

        for idx, blk in enumerate(self.blocks):
            blk.T, blk.H, blk.W = T, H, W
            if idx%2 == 0:
                x, v1, k1, q1 = blk(x, attn_mask)
            else:
                x, v2, k2, q2 = blk(x, attn_mask)

        if self.num_sem_blocks > 0:
            self.semantic_layer.T, self.semantic_layer.H, self.semantic_layer.W = T, H, W
            seg_map, x = self.semantic_layer(x)
        else:
            seg_map = None

        if self.downsample is not None:
            x_now = x
            x_now = self.norm(x_now)

            if C != 128:
                x_down = self.downsample(x, T, H, W)
                T, H, W = T // 2, H // 2, W // 2
            else:
                x_down = self.downsamplelast(x, T, H, W)
                T, H, W = T, H // 2, W // 2

        else:
            x = self.norm(x)
            x_now = x
            x_down = x_now

        return seg_map, x_now, x_down, T, H, W, v1, k1, q1, v2, k2, q2  ##x_now是经过norm的该层输出，x_down是未经过norm的传入下一阶段的



class MaskRadarBackbone(nn.Module):
    def __init__(self,
                 img_size=(16,128,128),
                 patch_size=(2,2,2),
                 in_chans=2,
                 num_cls=3,
                 embed_dim=64,
                 depths=[2, 2, 2],
                 num_heads=[2, 4, 8],
                 window_size=(4, 4, 4),
                 sem_window_size=(4, 4, 4),
                 num_sem_blocks=[1, 1, 1],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 frozen_stages=-1,
                 ape=True):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.num_cls = num_cls

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            pretrain_img_size = to_3tuple((16, 128, 128))
            patch_size = to_3tuple((2, 2, 2))
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1],
                                  pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)


        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                i=i_layer,
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                sem_window_size=sem_window_size,
                num_sem_blocks=num_sem_blocks[i_layer],
                num_cls=num_cls,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                shift_type='psm'
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)

            print(f'load model from: {self.pretrained}')

        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1,self.patch_size[0],1,1)/self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,
                                                                                                                   L2).permute(
                        1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        T, H, W = x.shape[2],x.shape[3],x.shape[4]
        x = x.flatten(2).transpose(1,2)
        x = self.pos_drop(x)

        outs = []
        cls_outs =[]
        v1_values = []
        k1_values = []
        q1_values = []
        v2_values = []
        k2_values = []
        q2_values = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            T1, H1, W1 = T, H, W
            cls_out, x_out, x, T, H, W, v1, k1, q1, v2, k2, q2 = layer(x, T, H, W)

            out = x_out.view(-1, T1, H1, W1, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
            outs.append(out)

            cls_out = cls_out.view(-1, T1, H1, W1, self.num_cls).permute(0, 4, 1, 2, 3).contiguous()
            cls_outs.append(cls_out)

            v1_values.append(v1)
            k1_values.append(k1)
            q1_values.append(q1)
            v2_values.append(v2)
            k2_values.append(k2)
            q2_values.append(q2)

        B, L, C = x.shape
        x = x.view(B, C, T, H, W)

        return x, tuple(outs), cls_outs, v1_values, k1_values, q1_values, v2_values, k2_values, q2_values



if __name__ == '__main__':
    """
    Test MaskRadarBackbone
    """
    input = torch.rand((1, 2, 16, 128, 128), device='cpu')
    model = MaskRadarBackbone(
                 img_size=(16,128,128),
                 patch_size=(2,2,2),
                 in_chans=2,
                 num_cls=3,
                 embed_dim=64,
                 depths=[2, 2, 2],
                 num_heads=[2, 4, 8],
                 window_size=(4, 4, 4),
                 sem_window_size=(4, 4, 4),
                 num_sem_blocks=[1, 1, 1],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 frozen_stages=-1,
                 ape=False
)
    mainoutput, output1, output2, v1, k1, q1, v2, k2, q2= model(input)
    print(mainoutput.shape) #torch.Size([1, 256, 4, 16, 16])
    print(output1[0].shape) #torch.Size([1, 64, 8, 64, 64])
    print(output1[1].shape) #torch.Size([1, 128, 4, 32, 32])
    print(output1[2].shape) #torch.Size([1, 256, 4, 16, 16])
    print(output2[0].shape) #torch.Size([1, 3, 8, 64, 64])
    print(output2[1].shape) #torch.Size([1, 3, 4, 32, 32])
    print(output2[2].shape) #torch.Size([1, 3, 4, 16, 16])
    print("//////////")
    print(v1[0].shape)
    print(v1[1].shape)
    print(v1[2].shape)
    print("//////////")
    print(k1[0].shape)
    print(k1[1].shape)
    print(k1[2].shape)
    print("//////////")
    print(q1[0].shape)
    print(q1[1].shape)
    print(q1[2].shape)
    print("//////////")
    print(v2[0].shape)
    print(v2[1].shape)
    print(v2[2].shape)
    print("//////////")
    print(k2[0].shape)
    print(k2[1].shape)
    print(k2[2].shape)
    print("//////////")
    print(q2[0].shape)
    print(q2[1].shape)
    print(q2[2].shape)








