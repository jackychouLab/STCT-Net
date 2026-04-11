import torch
import torch.nn as nn



class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.conv1 = nn.Conv3d(2, 16, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.conv2 = nn.Conv3d(16, 32, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.conv3 = nn.Conv3d(32, 64, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.conv4 = nn.Conv3d(64, 128, (6, 4, 4), stride=2, padding=(2, 1, 1))
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x


class DetectNet(nn.Module):
    def __init__(self, n_class=1, win_size=16):
        super(DetectNet, self).__init__()
        self.win_size = win_size
        self.deconv1 = nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=(2, 2, 2))
        self.deconv2 = nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2))
        self.deconv3 = nn.ConvTranspose3d(16, n_class, (2, 2, 2), stride=(2, 2, 2))
        self.upsample = nn.Upsample(size=(16, 128, 128), mode='nearest')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        return x


class RODEncode_RA(nn.Module):
    def __init__(self):
        super(RODEncode_RA, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=2, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        return x


class RODDecode_RA(nn.Module):
    def __init__(self):
        super(RODDecode_RA, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.convt1(x))
        x = self.prelu(self.convt2(x))
        x = self.prelu(self.convt3(x))
        return x


class RODEncode_RV(nn.Module):
    def __init__(self):
        super(RODEncode_RV, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=1, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        return x


class RODDecode_RV(nn.Module):
    def __init__(self):
        super(RODDecode_RV, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.prelu(self.convt1(x))
        x = self.prelu(self.convt2(x))
        x = self.prelu(self.convt3(x))
        return x


class RODEncode_VA(nn.Module):
    def __init__(self):
        super(RODEncode_VA, self).__init__()
        self.conv1a = nn.Conv3d(in_channels=1, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(1, 1, 1), padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256,
                                kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2))
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        return x


class RODDecode_VA(nn.Module):

    def __init__(self):
        super(RODDecode_VA, self).__init__()
        self.convt1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                         kernel_size=(4, 6, 6), stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt3 = nn.ConvTranspose3d(in_channels=64, out_channels=32,
                                         kernel_size=(3, 6, 6), stride=(1, 2, 2), padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.prelu(self.convt1(x))
        x = self.prelu(self.convt2(x))
        x = self.prelu(self.convt3(x))
        return x


class Fuse_fea_new_rep(nn.Module):
    def __init__(self):
        super(Fuse_fea_new_rep, self).__init__()
        self.convt1 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2))
        self.convt2 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.convt3 = nn.Conv3d(in_channels=32, out_channels=16,
                                kernel_size=(3, 1, 21), stride=(1, 1, 1), padding=(0, 0, 0),
                                dilation=(1, 1, 6))
        self.convt4 = nn.Conv3d(in_channels=48, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_ra, feas_rv, feas_va):
        feas_rv = torch.sum(feas_rv, 4, keepdim=True)
        feas_ra1 = feas_rv.expand(-1, -1, -1, -1, 128)
        feas_va = torch.sum(feas_va, 4, keepdim=True)
        feas_va = torch.transpose(feas_va, 3, 4)
        feas_ra2 = feas_va.expand(-1, -1, -1, 128, -1)
        fea_shap = feas_ra.shape
        feas_ra = feas_ra.permute(0, 2, 1, 3, 4)
        feas_ra = torch.unsqueeze(torch.reshape(feas_ra, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2)
        feas_ra1 = feas_ra1.permute(0, 2, 1, 3, 4)
        feas_ra1 = torch.unsqueeze(torch.reshape(feas_ra1, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2)
        feas_ra2 = feas_ra2.permute(0, 2, 1, 3, 4)
        feas_ra2 = torch.unsqueeze(torch.reshape(feas_ra2, (-1, fea_shap[1], fea_shap[3], fea_shap[4])), 2)
        feas_ra = torch.cat((feas_ra, feas_ra1, feas_ra2), 2)
        x1 = torch.squeeze(self.prelu(self.convt1(feas_ra)))
        x2 = torch.squeeze(self.prelu(self.convt2(feas_ra)))
        feas_ra = torch.nn.functional.pad(feas_ra, (60, 60, 0, 0, 0, 0), "circular")
        x3 = torch.squeeze(self.prelu(self.convt3(feas_ra)))
        x1 = torch.cat((x1, x2, x3), 1)
        x = torch.transpose(torch.reshape(x1, (fea_shap[0], fea_shap[2], 48, fea_shap[3], fea_shap[4])), 1, 2)
        x = self.sigmoid(self.convt4(x))
        return x


class RAMP(nn.Module):
    def __init__(self, n_class=1, win_size=16):
        super(RAMP, self).__init__()
        self.backbone = BackboneNet()
        self.detect = DetectNet(n_class, win_size)
        self.c3d_encode_ra = RODEncode_RA()
        self.c3d_decode_ra = RODDecode_RA()
        self.c3d_encode_rv = RODEncode_RV()
        self.c3d_decode_rv = RODDecode_RV()
        self.c3d_encode_va = RODEncode_VA()
        self.c3d_decode_va = RODDecode_VA()
        self.fuse_fea = Fuse_fea_new_rep()

    def forward(self, x_ra, x_rv, x_va):
        x_ra = self.c3d_encode_ra(x_ra)
        feas_ra = self.c3d_decode_ra(x_ra)
        x_rv = self.c3d_encode_rv(x_rv)
        feas_rv = self.c3d_decode_rv(x_rv)
        x_va = self.c3d_encode_va(x_va)
        feas_va = self.c3d_decode_va(x_va)
        dets = self.fuse_fea(feas_ra, feas_rv, feas_va)
        dets2 = self.fuse_fea(torch.zeros_like(feas_ra), feas_rv, feas_va)
        return dets, dets2