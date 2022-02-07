'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
'''

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

_BATCH_NORM_DECAY = 0.1
_BATCH_NORM_EPSILON = 1e-5


def batch_norm2d(num_features):
    return nn.BatchNorm2d(num_features, eps=_BATCH_NORM_EPSILON, momentum=_BATCH_NORM_DECAY)


def _weights_init(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight)


class IdentityBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = batch_norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = batch_norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = batch_norm2d(planes)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = self.act(out)
        return out


class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = batch_norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = batch_norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = batch_norm2d(planes)
        self.conv_shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn_shortcut = batch_norm2d(planes)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.bn_shortcut(self.conv_shortcut(x))
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, config, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.padd = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = batch_norm2d(16)

        unzip_info = list(zip(*config))
        num_layers = unzip_info[0]
        filters = unzip_info[1]
        strides = unzip_info[2]
        self.layer1 = self._make_layer(num_layers[0], filters[0], strides[0])
        self.layer2 = self._make_layer(num_layers[1], filters[1], strides[1])
        self.layer3 = self._make_layer(num_layers[2], filters[2], strides[2])
        self.bn2 = batch_norm2d(self.in_planes)
        self.linear = nn.Linear(filters[2], num_classes)
        self.act = nn.ReLU()
        self.apply(_weights_init)

    def _make_layer(self, num_layers, planes, stride):
        layers = nn.ModuleList()
        layers.append(ConvBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for layer in range(num_layers - 1):
            layers.append(IdentityBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, debug=False):
        x = self.padd(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn2(out)
        out = F.avg_pool2d(out, out.size()[3])
        out_ = out.view(out.size(0), -1)
        out = self.linear(out_)
        if debug:
            return out_
        else:
            return out


def resnet32(num_classes=10):
    return ResNet([(5, 16, 1), (5, 32, 2), (5, 64, 2)], num_classes)



