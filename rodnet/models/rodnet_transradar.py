import torch
from torch import nn
from operator import itemgetter
import torchvision.ops



class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        assert type(kernel_size) == tuple or type(kernel_size) == int
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=self.padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=self.padding, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias, padding=self.padding, mask=modulator, stride=self.stride)
        return x


def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn, dim=64, kernel_size=3):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation
        self.deform = DeformableConv2d(dim, dim, (kernel_size, kernel_size), padding=(kernel_size // 2, kernel_size // 2))

    def forward(self, x, **kwargs):
        x = self.deform(x)
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape
        axial = axial.reshape(-1, t, d)
        axial = self.fn(axial, **kwargs)
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class ADA(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_heads=None, dim_index=1, deform_k=[3, 3, 3, 3, 3, 3, 3, 3]):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)
        self.pos_emb = nn.Identity()
        layers = nn.ModuleList([])
        for mb in range(depth):
            attn_functions = nn.ModuleList([PermuteToFrom(permutation,
                                                          PreNorm(dim, SelfAttention(dim, heads, dim_heads)), dim=dim,
                                                          kernel_size=deform_k[mb]) for permutation in permutations])
            layers.append(attn_functions)

        self.layers = Sequential(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock_ra(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(k_size[0] + 1, k_size[1], k_size[2]), padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class EncodingBranch(nn.Module):
    def __init__(self, signal_type, k_size = 3, in_c=2):
        super().__init__()
        self.signal_type = signal_type
        if self.signal_type == 'range_angle':
            self.double_3dconv_block1 = Double3DConvBlock_ra(in_ch=in_c, out_ch=128, k_size=(k_size, 3, 3), pad=(0, 1, 1), dil=1)
        else:
            self.double_3dconv_block1 = Double3DConvBlock(in_ch=in_c, out_ch=128, k_size=(k_size, 3, 3), pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)
        x1_down = self.max_pool(x1)
        x2 = self.double_conv_block2(x1_down)
        x2_down = self.max_pool(x2)
        x3 = self.single_conv_block1_1x1(x2_down)
        return x3


class TransRad(nn.Module):
    def __init__(self, n_classes, n_chirps, deform_k = [3, 3, 3, 3, 3, 3, 3, 3], depth = 8, channels = 64):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_chirps
        self.rd_encoding_branch = EncodingBranch('range_doppler', k_size = 1, in_c=2)
        self.ra_encoding_branch = EncodingBranch('range_angle', k_size = n_chirps//2, in_c=2)
        self.ad_encoding_branch = EncodingBranch('angle_doppler', k_size = 1, in_c=2)
        self.pre_trans1 = ConvBlock(128*3,(128*3)//2,1,0,1)
        self.pre_trans2 = ConvBlock((128*3)//2,channels,1,0,1)
        self.ADA = ADA(dim=channels, depth = depth, deform_k = deform_k)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=channels, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_upconv1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3, pad=1, dil=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)


    def forward(self, x_rd, x_ra, x_ad):
        x_rd = torch.permute(x_rd, [0, 4, 1, 2, 3])
        x_ad = torch.permute(x_ad, [0, 4, 1, 2, 3])
        x_ra = torch.squeeze(x_ra, 2)
        ra_latent = self.ra_encoding_branch(x_ra)
        rd_latent = self.rd_encoding_branch(x_rd)
        ad_latent = self.ad_encoding_branch(x_ad)
        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        x3 = self.pre_trans2(self.pre_trans1(x3))
        x3 = self.ADA(x3)
        x4_ra = self.ra_single_conv_block2_1x1(x3)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_ra = self.ra_double_conv_block1(x5_ra)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_ra = self.ra_double_conv_block2(x7_ra)
        x9_ra = self.ra_final(x8_ra)
        x9_ra = torch.unsqueeze(x9_ra, 2)
        return x9_ra