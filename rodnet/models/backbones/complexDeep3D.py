import torch
from torch import nn
from torch.nn import functional as F



class ComplexStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ComplexStem, self).__init__()
        self.real_conv = nn.Conv3d(in_channels, out_channels, (1, 4, 4), (1, 4, 4), (0, 0, 0), bias=False)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, (1, 4, 4), (1, 4, 4), (0, 0, 0), bias=False)
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
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ComplexConv3d, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.real_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=padding, groups=groups, bias=False)
        self.bias = bias
        if self.bias:
            self.real_para = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
            self.imag_para = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        # B C F R A real,imag
        real = x[..., 0]
        imag = x[..., 1]
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        if self.bias:
            real_out = real_out + self.real_para
            imag_out = imag_out + self.imag_para

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
            Cri = self.running_covar[:, 2] # + self.eps

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
    def __init__(self, channels):
        super(ComplexUpConv3d, self).__init__()
        self.real_conv = nn.ConvTranspose3d(channels, channels, 2, stride=2, padding=0, bias=False)
        self.imag_conv = nn.ConvTranspose3d(channels, channels, 2, stride=2, padding=0, bias=False)
        self.norm = ComplexBatchNorm3D(channels)
        self.act = ComplexRELU()

    def forward(self, x):
        # B C F R A real,imag
        real = x[..., 0]
        imag = x[..., 1]
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        out = torch.stack((real_out, imag_out), dim=-1)

        return self.act(self.norm(out))


class ComplexBlock(nn.Module):
    def __init__(self, dim, expansion=0.75, kernel_size=(3, 3, 3)):
        super(ComplexBlock, self).__init__()

        self.gconv = ComplexConv3d(dim, dim, kernel_size, groups=dim, bias=False)
        self.gnorm = ComplexBatchNorm3D(dim)
        self.pwconv1 = ComplexConv3d(dim, int(dim * expansion), (1, 1, 1), groups=1, bias=True)
        self.pwconv2 = ComplexConv3d(int(dim * expansion), dim, (1, 1, 1), groups=1, bias=False)
        self.norm = ComplexBatchNorm3D(dim)
        self.act = ComplexRELU()

    def forward(self, x):
        shortcut = x
        x = self.act((self.gnorm(self.gconv(x))))
        x = self.act(self.pwconv1(x))
        x = self.norm(self.pwconv2(x))

        return self.act(x + shortcut)


class ComplexHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ComplexHead, self).__init__()
        self.gconv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.gnorm_1 = nn.BatchNorm3d(in_channels)
        self.dconv_1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.dnorm_1 = nn.BatchNorm3d(in_channels)
        self.gconv_2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.gnorm_2 = nn.BatchNorm3d(in_channels)
        self.dconv_2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.dnorm_2 = nn.BatchNorm3d(in_channels)
        self.cconv = nn.Conv3d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.gnorm_1(self.gconv_1(x)))
        x = self.act(self.dnorm_1(self.dconv_1(x)))
        x = self.act(self.gnorm_2(self.gconv_2(x)))
        x = self.act(self.dnorm_2(self.dconv_2(x)))
        x = self.cconv(x)

        return x

class ComplexFPN(nn.Module):
    def __init__(self, input_dims, out_channels, train_type='single'):
        super(ComplexFPN, self).__init__()
        self.train_type = train_type
        self.input_convs = nn.ModuleList()
        for i in range(len(input_dims)):
            self.input_convs.append(nn.Sequential(
                ComplexConv3d(input_dims[i], out_channels, (1, 1, 1), groups=1, bias=False),
                ComplexBatchNorm3D(out_channels),
                ComplexRELU()
            ))
        if self.train_type == 'single':
            self.output_conv = nn.Sequential(
                ComplexConv3d(out_channels, out_channels, (3, 3, 3), groups=1, bias=False),
                ComplexBatchNorm3D(out_channels),
                ComplexRELU()
            )
        else:
            self.output_convs = nn.ModuleList()
            for i in range(len(input_dims)):
                self.output_convs.append(nn.Sequential(
                    ComplexConv3d(out_channels, out_channels, (3, 3, 3), groups=1, bias=False),
                    ComplexBatchNorm3D(out_channels),
                    ComplexRELU()
                ))
        self.up_convs = nn.ModuleList()
        for i in range(len(input_dims) - 1):
            self.up_convs.append(ComplexUpConv3d(out_channels))

    def forward(self, xs):
        outs = []
        for i in range(len(self.input_convs)):
            outs.append(self.input_convs[i](xs[i]))

        for i in range(len(outs) - 1, 0, -1):
            outs[i - 1] += self.up_convs[i-1](outs[i])

        if self.train_type == 'single':
            outs[0] = self.output_conv(outs[0])
            return [outs[0]]
        else:
            for i in range(len(self.input_convs)):
                outs[i] = self.output_convs[i](outs[i])
            return outs


class ComplexDoppler(nn.Module):
    def __init__(self, dims):
        super(ComplexDoppler, self).__init__()
        self.convDowns = nn.ModuleList()
        for i in range(1, len(dims)):
            self.convDowns.append(ComplexDownConv3d(dims[i - 1], dims[i]))

    def forward(self, x):
        results = []
        x_real = x[..., 0].type(torch.complex64)
        x_imag = x[..., 1].type(torch.complex64)
        x_complex = x_real + 1j * x_imag
        doppler = torch.fft.fft(x_complex, dim=1)
        doppler_real = doppler.real.type(torch.float32)
        doppler_imag = doppler.imag.type(torch.float32)
        doppler = torch.stack([doppler_real, doppler_imag], dim=-1)
        results.append(doppler)
        for i in range(len(self.convDowns)):
            results.append(self.convDowns[i](results[i]))

        return results


class ComplexNet(nn.Module):

    def __init__(self, in_channels, depths, dims, num_classes, expansion=0.75, use_mse_loss=False, train_type='single', add_doppler=False):
        super(ComplexNet, self).__init__()
        assert len(depths) == len(dims)
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            ComplexStem(in_channels, dims[0])
        )
        self.train_type = train_type
        self.add_doppler = add_doppler
        if self.add_doppler:
            self.doppler = ComplexDoppler(dims)
            self.doppler_conv = nn.ModuleList()
        for i in range(len(depths) - 1):
            self.downsample_layers.append(ComplexDownConv3d(dims[i], dims[i + 1]))
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ComplexBlock(dim=dims[i], expansion=expansion)
                  for _ in range(depths[i])],
            )
            self.stages.append(stage)
            if self.add_doppler:
                self.doppler_conv.append(
                    nn.Sequential(
                        ComplexConv3d(dims[i], dims[i], (1, 1, 1), groups=1, bias=False),
                        ComplexBatchNorm3D(dims[i]),
                        ComplexRELU()
                    )
                )
        self.decoder = ComplexFPN(dims, dims[1], train_type=self.train_type)
        self.head = ComplexHead(dims[1] * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose3d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # B real&imag F chirps R A --> B chirps F R A real&imag
        x = x.permute(0, 3, 2, 4, 5, 1).contiguous()
        fs = []
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            if i == 0 and self.add_doppler:
                dopplers = self.doppler(x)
            x = self.stages[i](x)
            if self.add_doppler:
                x = x + dopplers[i]
                x = self.doppler_conv[i](x)
            fs.append(x)

        outs = self.decoder(fs)

        if self.train_type == "single":
            real = outs[0][..., 0]
            imag = outs[0][..., 1]
            mag = torch.clamp(torch.sqrt(real ** 2 + imag ** 2), min=1e-5)
            phase = torch.atan2(imag, real)
            input_f = torch.concat((mag, phase), dim=1)
            result = self.head(input_f)
            if not self.use_mse_loss:
                result = self.sigmoid(result)
            return [result], None

        if self.train_type == "multi":
            pros = []
            results = []
            for i in range(len(outs)):
                real = outs[i][..., 0]
                imag = outs[i][..., 1]
                mag = torch.clamp(torch.sqrt(real ** 2 + imag ** 2), min=1e-5)
                phase = torch.atan2(imag, real)
                result = self.head(torch.concat((mag, phase), dim=1))
                if not self.use_mse_loss:
                    result = self.sigmoid(result)
                results.append(result)
            return results, pros


def ComplexNet_t(in_channels: int, num_classes: int, use_mse_loss: bool, train_type: str):
    model = ComplexNet(in_channels=in_channels, depths=[3, 6, 3], dims=[32, 64, 128], num_classes=num_classes, use_mse_loss=use_mse_loss, train_type=train_type)
    return model


def ComplexNet_s(in_channels: int, num_classes: int, use_mse_loss: bool, train_type: str):
    model = ComplexNet(in_channels=in_channels, depths=[3, 6, 3], dims=[64, 128, 256], num_classes=num_classes, use_mse_loss=use_mse_loss, train_type=train_type)
    return model