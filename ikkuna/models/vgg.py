'''
.. moduleauthor Rasmus Diederichsen

This module defines :class:`ikkuna.models.VGG`, a small VGG network
'''
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, input_shape, num_classes=1000, exporter=None):
        super(VGG, self).__init__()
        self.in_channels = input_shape[-1]
        assert 1 <= self.in_channels <= 3, 'Expected 1-3 channels'
        assert input_shape[0] >= 32 and input_shape[1] >= 32, 'Images too small (min 32px per side)'

        self.features = self._make_layers()
        feature_outputs = 512 * (input_shape[0] // 2**5)  * (input_shape[1] // 2**5)
        self.classifier = nn.Linear(feature_outputs, num_classes)

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
        in_channels = self.in_channels
        for x in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=False)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
