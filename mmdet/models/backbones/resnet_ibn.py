from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from .pooling import build_pooling_layer
from ..builder import BACKBONES
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a

@BACKBONES.register_module()
class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, out_indices=(0, 1, 2, 3), pooling_type='avg'):

        print('pooling_type: {}'.format(pooling_type))
        super(ResNetIBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.out_indices = out_indices

        self.resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        self.resnet.layer4[0].conv2.stride = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)
        self.res_layers = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]

        if not pretrained:
            self.reset_params()
    
    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = layer_name(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)