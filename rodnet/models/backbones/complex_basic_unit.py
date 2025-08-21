import torch
from torch import nn
from torch.nn import functional as F



class ComplexStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=None, stride=None, padding=None):
        super(ComplexStem, self).__init__()
        kernel_size = 7 if kernel_size is None else kernel_size
        stride = 1 if stride is None else stride
        padding = 3 if padding is None else padding
        self.real_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = ComplexBatchNorm3D(out_channels)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]
        real_out = self.real_conv(x_real) - self.imag_conv(x_imag)
        imag_out = self.real_conv(x_imag) + self.imag_conv(x_real)
        x = torch.stack((real_out, imag_out), dim=-1)
        x = self.norm(x)

        return x


class ComplexConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=None, groups=1, bias=False):
        super(ComplexConv3d, self).__init__()
        stride = 1 if stride is None else stride
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2) if padding is None else padding
        self.real_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x):
        # B C F R A real,imag
        real = x[..., 0]
        imag = x[..., 1]
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)

        return torch.stack((real_out, imag_out), dim=-1)


class ComplexBatchNorm3D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm3D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,3))
            self.bias = nn.Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_real', torch.zeros(num_features))
            self.register_buffer('running_mean_imag', torch.zeros(num_features))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_real', None)
            self.register_parameter('running_mean_imag', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean_real.zero_()
            self.running_mean_imag.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

        if self.affine:
            nn.init.constant_(self.weight[:, :2],1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)

    def forward(self, input):
        input_real = input[..., 0]
        input_imag = input[..., 1]
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            mean_real = input_real.mean([0, 2, 3, 4])
            mean_imag = input_imag.mean([0, 2, 3, 4])
        else:
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean_real = (1 - exponential_average_factor) * self.running_mean_real + exponential_average_factor * mean_real
                self.running_mean_imag = (1 - exponential_average_factor) * self.running_mean_imag + exponential_average_factor * mean_imag

        input_real = input_real - mean_real[None, :, None, None, None]
        input_imag = input_imag - mean_imag[None, :, None, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input_real.numel() / input_real.size(1)
            Crr = 1. / n * input_real.pow(2).sum(dim=[0, 2, 3, 4]) + self.eps
            Cii = 1. / n * input_imag.pow(2).sum(dim=[0, 2, 3, 4]) + self.eps
            Cri = (input_real.mul(input_imag)).mean(dim=[0, 2, 3, 4])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        norm_real = Rrr[None, :, None, None, None] * input_real + Rri[None, :, None, None, None] * input_imag
        norm_imag = Rii[None, :, None, None, None] * input_imag + Rri[None, :, None, None, None] * input_real

        if self.affine:
            real = self.weight[None, :, 0, None, None, None] * norm_real + self.weight[None, :, 2, None, None, None] * norm_imag + self.bias[None, :, 0, None, None, None]
            imag = self.weight[None, :, 2, None, None, None] * norm_real + self.weight[None, :, 1, None, None, None] * norm_imag + self.bias[None, :, 1, None, None, None]
            norm_real = real
            norm_imag = imag

        return torch.stack((norm_real, norm_imag), dim=-1)


class ComplexRELU(nn.Module):
    def __init__(self):
        super(ComplexRELU, self).__init__()

    def forward(self, x):
        real = F.relu(x[..., 0])
        imag = F.relu(x[..., 1])

        return torch.stack((real, imag), dim=-1)


class ComplexDownConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexDownConv3d, self).__init__()
        self.real_conv = nn.Conv3d(in_channels, out_channels, 2, stride=2, padding=0, bias=False)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, 2, stride=2, padding=0, bias=False)
        self.norm = ComplexBatchNorm3D(out_channels)
        self.act = ComplexRELU()

    def forward(self, x):
        # B C F R A real,imag
        real = x[..., 0]
        imag = x[..., 1]
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        out = torch.stack((real_out, imag_out), dim=-1)

        return self.act(self.norm(out))


class ComplexUpConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=None, stride=None, padding=None, bias=False):
        super(ComplexUpConv3d, self).__init__()
        kernel_size = 2 if kernel_size is None else kernel_size
        stride = 2 if stride is None else stride
        padding = 0 if padding is None else padding
        self.real_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.imag_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        # B C F R A real,imag
        real = x[..., 0]
        imag = x[..., 1]
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        out = torch.stack((real_out, imag_out), dim=-1)

        return out