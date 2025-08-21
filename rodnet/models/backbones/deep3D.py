import torch.nn as nn
import torch
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim, expansion=1., kernel_size=(3, 3, 3), stride=(1, 1, 1), layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, groups=dim)
        self.norm1 = nn.GroupNorm(8, dim)
        self.pwconv1 = nn.Linear(dim, int(dim * expansion))
        self.pwconv2 = nn.Linear(int(dim * expansion), dim)
        self.norm2 = nn.GroupNorm(8, dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 4, 1)  # B C F R A -- > B F R A C
        x = self. pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)  # B F R A C -- > B C F R A
        x = self.norm2(x)
        x = self.act(x + shortcut)
        return x


class downSampler(nn.Module):
    def __init__(self, input_dim, out_dim, is_stem=False):
        super(downSampler, self).__init__()
        self.is_stem = is_stem
        if self.is_stem:
            self.conv = nn.Conv3d(input_dim, out_dim, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=True)
        else:
            self.conv = nn.Conv3d(input_dim, out_dim, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True)
        self.norm = nn.GroupNorm(8, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        if self.is_stem:
            x = self.norm(x)
        else:
            x = self.act(self.norm(x))
        return x


class Conv3D_with_GN_act(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size):
        super(Conv3D_with_GN_act, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.conv = nn.Conv3d(input_dim, out_dim, kernel_size, stride=(1, 1, 1), padding=padding, bias=True)
        self.norm = nn.GroupNorm(8, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x


class FPN_Z(nn.Module):
    def __init__(self, input_dims, out_dim, train_type='single'):
        super(FPN_Z, self).__init__()
        self.train_type = train_type
        self.input_convs = nn.ModuleList()
        if self.train_type == 'single':
            self.output_conv = Conv3D_with_GN_act(out_dim, out_dim, kernel_size=(3, 3, 3))
        else:
            self.output_convs = nn.ModuleList()
            for i in range(len(input_dims)):
                self.output_convs.append(Conv3D_with_GN_act(out_dim, out_dim, kernel_size=(3, 3, 3)))
        for i in range(len(input_dims)):
            self.input_convs.append(Conv3D_with_GN_act(input_dims[i], out_dim, kernel_size=(1, 1, 1)))


    def forward(self, xs):
        outs = []
        for i in range(len(self.input_convs)):
            outs.append(self.input_convs[i](xs[i]))

        for i in range(len(outs) - 1, 0, -1):
            prev_shape = outs[i - 1].shape[2:]
            outs[i - 1] =  outs[i - 1] + F.interpolate(outs[i], size=prev_shape, mode="trilinear", align_corners=False)

        if self.train_type == 'single':
            outs[0] = self.output_conv(outs[0])
            return [outs[0]]
        else:
            for i in range(len(self.input_convs)):
                outs[i] = self.output_convs[i](outs[i])
            return outs


class Deep3Dv1(nn.Module):

    def __init__(self, in_channels, depths, dims, num_classes, expansion=0.5, use_mse_loss=False, train_type='single'):
        super(Deep3Dv1, self).__init__()
        assert len(depths) == len(dims)
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(downSampler(in_channels, dims[0], True))
        self.train_type = train_type
        for i in range(len(depths) - 1):
            self.downsample_layers.append(downSampler(dims[i], dims[i + 1]))
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], expansion=expansion)
                  for _ in range(depths[i])],
            )
            self.stages.append(stage)
        self.decoder = FPN_Z(dims, dims[1], train_type=self.train_type)
        self.head = nn.Conv3d(dims[1], num_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss
        if 'mix' in self.train_type:
            self.projection = nn.Linear(dims[1], 1, bias=True)
            assert 'softmax' in self.train_type or 'sigmoid' in self.train_type
        if 'softmax' in self.train_type:
            self.softmax = nn.Softmax(dim=1)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.flatten = nn.Flatten()
        if 'sigmoid' in self.train_type:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.flatten = nn.Flatten()


    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fs = []

        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            fs.append(x)

        outs = self.decoder(fs)

        if self.train_type == "single":
            outs[0] = self.head(outs[0])
            if not self.use_mse_loss:
                outs[0] = self.sigmoid(outs[0])
            return outs, None

        if self.train_type == "multi":
            pros = []
            for i in range(len(outs)):
                outs[i] = self.head(outs[i])
                if not self.use_mse_loss:
                    outs[i] = self.sigmoid(outs[i])
            return outs, pros

        if 'mix' in self.train_type:
            pros = []
            for i in range(len(outs)):
                tf = self.pool(outs[i])
                outs[i] = self.head(outs[i])
                if 'sigmoid' in self.train_type:
                    pros.append(self.sigmoid(self.projection(self.flatten(tf))))
                if 'softmax' in self.train_type:
                    pros.append(self.projection(self.flatten(tf)))
                if not self.use_mse_loss:
                    outs[i] = self.sigmoid(outs[i])
            if 'softmax' in self.train_type:
                pros = torch.concat(pros, dim=1)
                pros = self.softmax(pros)
                pros = torch.chunk(pros, 3, 1)

            return outs, pros


def Deep3Dv1_t(in_channels: int, num_classes: int, use_mse_loss: bool, train_type: str):
    model = Deep3Dv1(in_channels=in_channels, depths=[1, 1, 2, 1], dims=[64, 128, 256, 512], num_classes=num_classes, use_mse_loss=use_mse_loss, train_type=train_type)
    return model


def Deep3Dv1_s(in_channels: int, num_classes: int, use_mse_loss: bool, train_type: str):
    model = Deep3Dv1(in_channels=in_channels, depths=[3, 3, 6, 3], dims=[128, 256, 512, 1024], num_classes=num_classes, use_mse_loss=use_mse_loss, train_type=train_type)
    return model

def Deep3Dv1_m(in_channels: int, num_classes: int, use_mse_loss: bool, train_type: str):
    model = Deep3Dv1(in_channels=in_channels, depths=[3, 3, 12, 3], dims=[128, 256, 512, 1024], num_classes=num_classes, use_mse_loss=use_mse_loss, train_type=train_type)
    return model

def Deep3Dv1_l(in_channels: int, num_classes: int, use_mse_loss: bool, train_type: str):
    model = Deep3Dv1(in_channels=in_channels, depths=[3, 3, 18, 3], dims=[128, 256, 512, 1024], num_classes=num_classes, use_mse_loss=use_mse_loss, train_type=train_type)
    return model