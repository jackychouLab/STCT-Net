import torch.nn as nn
import torch.nn.functional as F
import torch
from .rcrosnet_exo.adaptive_directional_attention import ADA
from .rcrosnet_exo.dab_transformer import build_transformer
from .rcrosnet_exo.position_encoding import build_position_encoding
from .rcrosnet_exo.rc_dab_transformer import build_transformer as build_transformer_rc
from .rcrosnet_exo.rc_position_encoding import build_position_encoding as build_position_encoding_rc



class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

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
    """ (3D conv => BN => LeakyReLU) * 2 """

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
    """ (2D conv => BN => LeakyReLU) """

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
    """
    Encoding branch for a single radar view.
    Same implementation as the original MVRSS paper.

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type, k_size = 3):
        super().__init__()
        self.signal_type = signal_type
        if self.signal_type == 'range_angle':
            self.double_3dconv_block1 = Double3DConvBlock_ra(in_ch=2, out_ch=64, k_size=(k_size, 3, 3), pad=(0, 1, 1), dil=1)
        else:
            self.double_3dconv_block1 = Double3DConvBlock(in_ch=3, out_ch=64, k_size=(k_size, 3, 3), pad=(0, 1, 1), dil=1)
        self.doppler_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.double_conv_block2 = DoubleConvBlock(in_ch=64, out_ch=128, k_size=3, pad=1, dil=1)
        self.down_conv_block1_1x1_1 = nn.Conv2d(128, 128, kernel_size=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.double_conv_block3 = DoubleConvBlock(in_ch=128, out_ch=256, k_size=3,pad=1, dil=1)
        self.down_conv_block1_1x1_2 = nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        if self.signal_type in ('image'):
            x1_down = self.max_pool2(x1)
        else:
            x1_down = self.max_pool(x1)

        x2 = self.double_conv_block2(x1_down)

        if self.signal_type in ('image'):
            x2_down = self.bn1(self.down_conv_block1_1x1_1(x2))
        else:
            x2_down = self.bn1(self.down_conv_block1_1x1_1(x2))

        x3 = self.double_conv_block3(x2_down)
        if self.signal_type in ('range_angle'):
            x4 = self.bn2(self.down_conv_block1_1x1_2(x3))

            return x3, x4
        else:
            x3 = self.bn2(self.down_conv_block1_1x1_2(x3))
            return x3

class MLP(nn.Module):
    # Modified from facebookresearch/detr
    # Very simple multi-layer perceptron (also called FFN)

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x

class FusionModule(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self):
        super().__init__()


        self.relu = nn.ReLU()

        expansion = 1
        num_queries = 6
        query_dim = 256
        pos_type = 'sine'
        drop_out = 0.1
        num_heads = 4
        enc_layers = 2
        dec_layers = 2
        dim_feedforward = 512
        pre_norm = False
        return_intermediate = True
        # dab_transformer
        hidden_dim = 64 * expansion
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.position_embedding = build_position_encoding_rc(hidden_dim=query_dim, position_embedding=pos_type)
        self.input_proj = nn.Conv2d(3, query_dim, kernel_size=1)
        self.transformer = build_transformer_rc(decoder = False,
                                              hidden_dim=query_dim,
                                              query_dim=hidden_dim,
                                              dropout=drop_out,
                                              nheads=num_heads,
                                              dim_feedforward=dim_feedforward,
                                              enc_layers=enc_layers,
                                              dec_layers=dec_layers,
                                              pre_norm=pre_norm,
                                              return_intermediate_dec=return_intermediate)
        self.para_embed = MLP(query_dim, query_dim, hidden_dim, 3)
        # self.para_embed = nn.Linear(query_dim, hidden_dim)
        self.query_dim = hidden_dim


    def forward(self, radar, image_to_ra):

        main_branch = (radar)
        main_branch2 = self.input_proj(image_to_ra)

        b, c, h, w = main_branch.shape

        padding_masks = torch.zeros((main_branch.shape[0], main_branch.shape[2], main_branch.shape[3]), dtype=torch.bool, device=main_branch.device)
        Pos = self.position_embedding(main_branch, padding_masks)  # 4 4 20 50
        memory = self.transformer(main_branch, main_branch2, padding_masks, self.query_embed.weight, Pos)

        fusion = memory.permute(1, 2, 0).view(b, c, h, w)

        return fusion

class GenerateRangeAngle(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        expansion = 1
        num_queries = 6
        query_dim = 256
        pos_type = 'sine'
        drop_out = 0.1
        num_heads = 4
        enc_layers = 2
        dec_layers = 2
        dim_feedforward = 512
        pre_norm = False
        return_intermediate = True
        # dab_transformer
        hidden_dim = 16 * expansion
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.position_embedding = build_position_encoding(hidden_dim=query_dim, position_embedding=pos_type)
        self.input_proj = nn.Conv2d(128, query_dim, kernel_size=1)
        self.transformer = build_transformer(hidden_dim=query_dim,
                                              query_dim=hidden_dim,
                                              dropout=drop_out,
                                              nheads=num_heads,
                                              dim_feedforward=dim_feedforward,
                                              enc_layers=enc_layers,
                                              dec_layers=dec_layers,
                                              pre_norm=pre_norm,
                                              return_intermediate_dec=return_intermediate)
        self.para_embed = MLP(query_dim, query_dim, hidden_dim, 3)
        # self.para_embed = nn.Linear(query_dim, hidden_dim)
        self.query_dim = hidden_dim

    def get_range_dist(self, x, eps=1e-20):
        return x.softmax(dim=3)

    def get_angle_dist(self, x, eps=1e-20):
        return x.softmax(dim=4)

    def forward(self, image):

        main_branch = (image)

        padding_masks = torch.zeros((main_branch.shape[0], main_branch.shape[2], main_branch.shape[3]), dtype=torch.bool, device=main_branch.device)
        Pos = self.position_embedding(main_branch, padding_masks)  # 4 4 20 50
        hs, reference = self.transformer(main_branch, padding_masks, self.query_embed.weight, Pos)

        tmp = self.para_embed(hs)
        tmp[..., :self.query_dim] += reference
        tmp_params = tmp[-1]

        #process
        range_params = tmp_params[:, :3, :].unsqueeze(3)
        angle_params = tmp_params[:, 3:, :].unsqueeze(2)

        generate_ra = (range_params * angle_params)

        return generate_ra



class RCRODNet(nn.Module):
    def __init__(self, n_classes, n_frames, deform_k = [3, 3, 3, 3, 3, 3, 3, 3], depth = 8, channels = 64):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.ra_encoding_branch = EncodingBranch('range_angle', k_size=n_frames//2)
        self.image_encoding_branch = EncodingBranch('image', k_size=1)

        self.generate_rangeangle = GenerateRangeAngle()
        self.FusionModule = FusionModule()

        self.pre_trans1 = ConvBlock(384, 192, 3, 1, 1)
        self.pre_trans2 = ConvBlock(192, 64, 1, 0, 1)
        self.ADA = ADA(dim=channels, depth=depth, deform_k=deform_k)

        # Decoding
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=channels, out_ch=128, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.ra_upconv0 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.ra_upconv1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=64, k_size=3,pad=1, dil=1)

        self.ra_upconv2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=64, out_ch=64, k_size=3, pad=1, dil=1)

        # Final 1D convs
        self.ra_final = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)

    def forward(self,image, x_ra):
        x_ra = x_ra.squeeze(2)
        image_latent = self.image_encoding_branch(image)
        ra_latent, ra_latent2 = self.ra_encoding_branch(x_ra)

        pesudo_ra = self.generate_rangeangle(image_latent)
        fusion =  self.FusionModule(ra_latent2, pesudo_ra)
        fusion = self.ra_upconv0(fusion)

        x3 = torch.cat((ra_latent, fusion), 1)
        x3 = self.pre_trans2(self.pre_trans1(x3))
        x3 = self.ADA(x3)

        x4_ra = self.ra_single_conv_block2_1x1(x3)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_ra = self.ra_double_conv_block1(x5_ra)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_ra = self.ra_double_conv_block2(x7_ra)
        x9_ra = self.ra_final(x8_ra)
        x9_ra = x9_ra.unsqueeze(-3)

        return x9_ra