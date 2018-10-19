'''
MIT License

Copyright (c) 2017 liukuang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''I tried torchvision's model code initially, but that one does not deal with image sizes smaller
than imagnet. An alternative implementation is `here
<https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py>`_. I'm not sure how faithful
the reproduction is, but ultimately it shouldn't matter.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, BlockType, num_blocks, num_classes=10, exporter=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BlockType, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BlockType, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BlockType, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BlockType, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*BlockType.expansion, num_classes)

        exporter.set_model(self)
        exporter.add_modules(self)

    def _make_layer(self, BlockType, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BlockType(self.in_planes, planes, stride))
            self.in_planes = planes * BlockType.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(**kwargs):
    '''Constructs a ResNet-18 model.

    Parameters
    ----------
    kwargs  :   dict
                ResNet parameters
    '''
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(pretrained=False, **kwargs):
    '''Constructs a ResNet-34 model.

    Parameters
    ----------
    kwargs  :   dict
                ResNet parameters
    '''
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(pretrained=False, **kwargs):
    '''Constructs a ResNet-50 model.

    Parameters
    ----------
    kwargs  :   dict
                ResNet parameters
    '''
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(pretrained=False, **kwargs):
    '''Constructs a ResNet-101 model.

    Parameters
    ----------
    kwargs  :   dict
                ResNet parameters
    '''
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    '''Constructs a ResNet-152 model.

    Parameters
    ----------
    kwargs  :   dict
                ResNet parameters
    '''
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
