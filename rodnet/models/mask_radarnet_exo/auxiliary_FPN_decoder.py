import torch
import torch.nn as nn
import numpy as np



class FPN_decoder(nn.Module):
    def __init__(self, feature_strides, in_channels, channels, num_class):
        super().__init__()

        self.features_strides = feature_strides
        self.in_channels = in_channels
        self.channels = channels
        self.dropout = nn.Dropout(0.1)
        self.finalupsample = nn.Upsample(scale_factor=(2, 2, 2),
                                         mode='trilinear',
                                         align_corners=True)

        self.cls_head = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0]))
            )
            cls_head = []
            flag2 = 0
            for k in range(head_length):
                if feature_strides[i] != feature_strides[0]:
                    if feature_strides[i] != feature_strides[2] or flag2 == 1:
                        cls_head.append(
                            nn.Upsample(
                                scale_factor=(2,2,2),
                                mode = 'trilinear',
                                align_corners=True
                            )
                        )
                    else:
                        cls_head.append(
                            nn.Upsample(
                                scale_factor=(1,2,2),
                                mode = 'trilinear',
                                align_corners=True
                            )
                        )
                        flag2 = 1
            self.cls_head.append(nn.Sequential(*cls_head))



    def forward(self, input):
        cls = input
        cls_output = cls[0]
        for i in range(1,len(self.features_strides)):
            cls_output = cls_output + self.cls_head[i](cls[i])

        cls_output = self.finalupsample(cls_output)

        return cls_output


if __name__ == '__main__':
    """
        Test FPN_decoder
    """
    input1 = torch.rand((1, 3, 8, 64, 64), device='cpu')
    input2 = torch.rand((1, 3, 4, 32, 32), device='cpu')
    input3 = torch.rand((1, 3, 4, 16, 16), device='cpu')
    input = [input1, input2, input3]

    feature_strides = [2,4,8]
    in_channels = [64,128,256]
    channels = 96
    num_class =3

    model = FPN_decoder(feature_strides=feature_strides, in_channels=in_channels, channels=channels, num_class=num_class)
    output= model(input)
    print(output.shape)

