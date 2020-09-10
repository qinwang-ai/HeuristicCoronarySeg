import torch
from net.ResUnet import ResUNet
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义单个3D FCN
class Model(nn.Module):
    def __init__(self, training):
        super().__init__()
        self.training = training
        self.stage1 = ResUNet(training=True, inchannel=2)
        self.stage2 = ResUNet(training=True, inchannel=2+2)

    def forward(self, inputs):
        pred1 = self.stage1(inputs)
        inputs = torch.cat([inputs, pred1], dim=1)
        pred2 = self.stage2(inputs)
        if self.training:
            return pred1, pred2
        else:
            return pred2


def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


net = Model(training=True)
net.apply(init)

