'''
.. moduleauthor Rasmus Diederichsen

This module defines :class:`ikkuna.models.AlexNetMini`, a reduced version of AlexNet.
Adapted from
https://github.com/sukilau/Ziff-deep-learning/blob/master/3-CIFAR10-lrate/CIFAR10-lrate.ipynb
'''
import torch


class AlexNetMini(torch.nn.Module):     # Pytorch Sphinx-doc is buggy here, so use full path
    '''Reduced AlexNet (basically just a few conv layers with relu and
    max-pooling) which attempts to adapt to arbitrary input sizes, provided they are large enough to
    survive the strides and conv cutoffs.

    Attributes
    ---------
    features    :   torch.nn.Module
                    Convolutional module, extracting features from the input
    classifier  :   torch.nn.Module
                    Classifier with relu and dropout
    H_out   :   int
                Output height of the classifier
    W_out   :   int
                Output width of the classifier
    '''
    def __init__(self, input_shape, num_classes=1000, exporter=None):
        '''
        Parameters
        ----------
        input_shape :   tuple
                        H, W C (beware of channel order, is different from what you call the model
                        with)
        '''
        super(AlexNetMini, self).__init__()

        # if channel dim not present, add 1
        if len(input_shape) == 2:
            input_shape.append(1)
        H, W, C = input_shape

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(C, 4, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(4, 8, kernel_size=3, padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # for each conv or padding layer, apply the output size formula ((length - filter_size +
        # 2*padding)/ stride) + 1
        H_conv1    = (H - 3 + 2 * 0) // 1 + 1
        H_conv2    = (H_conv1 - 3 + 2 * 0) // 1 + 1
        self.H_out = (H_conv2 - 2 + 2 * 0) // 2 + 1

        W_conv1    = (W - 3 + 2 * 0) // 1 + 1
        W_conv2    = (W_conv1 - 3 + 2 * 0) // 1 + 1
        self.W_out = (W_conv2 - 2 + 2 * 0) // 2 + 1

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(8 * self.H_out * self.W_out, 16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(16, num_classes),
            # torch.nn.Softmax()
        )

        if exporter:
            exporter.set_model(self)
            exporter.add_modules(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 8 * self.H_out * self.W_out)
        x = self.classifier(x)
        return x
