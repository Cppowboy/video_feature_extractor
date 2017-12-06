# coding: utf-8

'''
Boundary-aware video captioning
'''

import torch
import torch.nn as nn
import torchvision.models as models

from args import resnet_checkpoint


class AppearanceEncoder(nn.Module):
    # 使用ResNet50作为视觉特征提取器
    def __init__(self):
        super(AppearanceEncoder, self).__init__()
        self.resnet = models.resnet152()
        self.resnet.load_state_dict(torch.load(resnet_checkpoint))
        del self.resnet.fc
        del self.resnet.avgpool

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # x = self.resnet.avgpool(x)
        # x = x.view(x.size(0), -1)
        return x
