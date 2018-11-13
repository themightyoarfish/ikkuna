'''
.. moduleauthor Rasmus Diederichsen

This module defines :class:`ikkuna.models.VGG`, a small VGG network
'''
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, input_shape, num_classes=1000, exporter=None):
        super(VGG, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(512, num_classes)

        if exporter:
            exporter.set_model(self)
            exporter.add_modules(self)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        for x in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)