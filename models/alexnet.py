'''
.. module:: alexnet
.. moduleauthor Rasmus Diederichsen

This module defines :py:class:`AlexNetMini`, a reduced version of AlexNet. Adapted from
torchvision.models.alexnet.
'''
from ikkuna.export import Exporter
import torch.nn as nn


class AlexNetMini(nn.Module):
    '''Reduced AlexNet (basically just a few conv layers with relu and
    max-pooling) which attempts to adapt to arbitrary input sizes, provided they are large enough to
    survive the strides and conv cutoffs.

    Attributes
    ---------
    features    :   nn.Module
                    Convolutional module, extracting features from the input
    classifier  :   nn.Module
                    Linear classifier with relu and dropout
    H_out   :   int
                Output height of the classifier
    W_out   :   int
                Output width of the classifier
    '''

    def __init__(self, input_shape, num_classes=1000, exporter=None):
        super(AlexNetMini, self).__init__()
        self._exporter = e = exporter or Exporter()
        e.set_model(self)

        # if batch dim not present, add 1
        if len(input_shape) == 2:
            input_shape.append(1)
        H, W, C = input_shape

        self.features = nn.Sequential(
            (nn.Conv2d(C, 64, kernel_size=5, stride=2, padding=1)),
            (nn.ReLU(inplace=True)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            (nn.Conv2d(64, 192, kernel_size=3, padding=2)),
            (nn.ReLU(inplace=True)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            (nn.Conv2d(192, 192, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
        )
        self.H_out =  H // (2 * 2 * 2)
        self.W_out =  W // (2 * 2 * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            (nn.Linear(192 * self.H_out * self.W_out, 2048)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            (nn.Linear(2048, 2048)),
            nn.ReLU(inplace=True),
            e(nn.Linear(2048, num_classes)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 192 * self.H_out * self.W_out)
        x = self.classifier(x)
        return x
