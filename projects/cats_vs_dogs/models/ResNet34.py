# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:39:04 2019

@author: Snake
"""


from .BasicModule import BasicModule
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    # 实现子 module: Residual Block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        
        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(outchannel)
                )
        self.right = shortcut
        
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
        

class ResNet34(BasicModule):
    # 实现主 module： ResNet34
    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'
        self.num_classes = num_classes
        
        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
                )
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构建多个layer, 包含多个 residual block
        shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel)
                )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        