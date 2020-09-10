import torch
import torch.nn as nn
import torch.nn.functional as F

num_organ = 1
dropout_rate = 0.5
Max_ratio = 0.2


# 定义单个3D FCN
class ResUNet(nn.Module):

    def __init__(self, training, inchannel):
        super().__init__()

        self.training = training
        self.gf = 32
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, self.gf, 3, 1, padding=5, dilation=5),
            nn.PReLU(self.gf),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(self.gf * 2, self.gf * 2, 3, 1, padding=1),
            nn.PReLU(self.gf * 2),

            nn.Conv3d(self.gf * 2, self.gf * 2, 3, 1, padding=4, dilation=4),
            nn.PReLU(self.gf * 2),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(self.gf * 4, self.gf * 4, 3, 1, padding=1),
            nn.PReLU(self.gf * 4),

            nn.Conv3d(self.gf * 4, self.gf * 4, 3, 1, padding=1),
            nn.PReLU(self.gf * 4),

            nn.Conv3d(self.gf * 4, self.gf * 4, 3, 1, padding=3, dilation=3),
            nn.PReLU(self.gf * 4),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(self.gf * 8, self.gf * 8, 3, 1, padding=1),
            nn.PReLU(self.gf * 8),

            nn.Conv3d(self.gf * 8, self.gf * 8, 3, 1, padding=1),
            nn.PReLU(self.gf * 8),

            nn.Conv3d(self.gf * 8, self.gf * 8, 3, 1, padding=1),
            nn.PReLU(self.gf * 8),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(self.gf * 8, self.gf * 16, 3, 1, padding=1),
            nn.PReLU(self.gf * 16),

            nn.Conv3d(self.gf * 16, self.gf * 16, 3, 1, padding=1),
            nn.PReLU(self.gf * 16),

            nn.Conv3d(self.gf * 16, self.gf * 16, 3, 1, padding=1),
            nn.PReLU(self.gf * 16),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(self.gf * 8 + self.gf * 4, self.gf * 8, 3, 1, padding=1),
            nn.PReLU(self.gf * 8),

            nn.Conv3d(self.gf * 8, self.gf * 8, 3, 1, padding=1),
            nn.PReLU(self.gf * 8),

            nn.Conv3d(self.gf * 8, self.gf * 8, 3, 1, padding=1),
            nn.PReLU(self.gf * 8),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(self.gf * 4 + self.gf * 2, self.gf * 4, 3, 1, padding=1),
            nn.PReLU(self.gf * 4),

            nn.Conv3d(self.gf * 4, self.gf * 4, 3, 1, padding=1),
            nn.PReLU(self.gf * 4),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(self.gf * 2 + self.gf, self.gf * 2, 3, 1, padding=1),
            nn.PReLU(self.gf * 2),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(self.gf, self.gf * 2, 2, 2),
            nn.PReLU(self.gf * 2)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(self.gf * 2, self.gf * 4, 2, 2),
            nn.PReLU(self.gf * 4)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(self.gf * 4, self.gf * 8, 2, 2),
            nn.PReLU(self.gf * 8)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(self.gf * 8, self.gf * 16, 3, 1, padding=1),
            nn.PReLU(self.gf * 16)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(self.gf * 16, self.gf * 8, 2, 2),
            nn.PReLU(self.gf * 8),
            nn.BatchNorm3d(self.gf * 8)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(self.gf * 8, self.gf * 4, 2, 2),
            nn.PReLU(self.gf * 4),
            nn.BatchNorm3d(self.gf * 4)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(self.gf * 4, self.gf * 2, 2, 2),
            nn.PReLU(self.gf * 2),
            nn.BatchNorm3d(self.gf * 2)
        )

        self.map = nn.Sequential(
            nn.Conv3d(self.gf * 2, num_organ, 1),
        )

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, dropout_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, dropout_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, dropout_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs_bottle = F.dropout(outputs, dropout_rate, self.training)

        short_range6 = self.up_conv2(outputs_bottle)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        result = self.map(outputs)

        return result


# 网络参数初始化函数
def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


net = ResUNet(training=True, inchannel=2)
net.apply(init)
