import torch.nn as nn
from timm.models.layers import to_3tuple
from .mask_radarnet_exo.Backbone import MaskRadarBackbone
from .mask_radarnet_exo.Main_decoder import Main_decoder
from .mask_radarnet_exo.auxiliary_FPN_decoder import FPN_decoder



class MaskRadar(nn.Module):
    def __init__(self, num_classes=3, embed_dim=64, win_size=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_class = num_classes
        self.window_size = to_3tuple(win_size)

        feature_strides = [2, 4, 8]
        in_channels = [64, 128, 256]
        channels = 96

        self.backbone = MaskRadarBackbone(
            img_size=(16, 128, 128),
            patch_size=(2, 2, 2),
            in_chans=2,
            num_cls=self.num_class,
            embed_dim=self.embed_dim,
            depths=[2, 2, 2],
            num_heads=[2, 4, 8],
            window_size=self.window_size,
            sem_window_size=self.window_size,
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

        self.main_decoder = Main_decoder(
            num_classes=self.num_class,
            embed_dim=self.embed_dim,
            depths_decoder=[2, 2, 2],
            num_heads=[2, 4, 8],
            window_size=self.window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1,
            ape=False
        )

        self.aux_decoder = FPN_decoder(
            feature_strides=feature_strides, in_channels=in_channels, channels=channels, num_class=num_classes
        )

    def forward(self, input):
        smallest_feature, feature_outputs, cls_outputs, prev_v1, prev_k1, prev_q1, prev_v2, prev_k2, prev_q2 = self.backbone(input)
        logits = self.main_decoder(smallest_feature, feature_outputs, prev_v1, prev_k1, prev_q1, prev_v2, prev_k2, prev_q2)
        if self.training:
            cls_logits = self.aux_decoder(cls_outputs)
            return logits, cls_logits
        return logits