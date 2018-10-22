'''
.. moduleauthor Rasmus Diederichsen

This module defines :class:`ikkuna.models.AlexNetMini`, a reduced version of AlexNet. Adapted from
:meth:`torchvision.models.alexnet()`.
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
        exporter.set_model(self)

        # if channel dim not present, add 1
        if len(input_shape) == 2:
            input_shape.append(1)
        H, W, C = input_shape

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(C, 64, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=3, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.H_out =  H // (2 * 2 * 2)
        self.W_out =  W // (2 * 2 * 2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(192 * self.H_out * self.W_out, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, num_classes),
        )

        exporter.add_modules(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 192 * self.H_out * self.W_out)
        x = self.classifier(x)
        return x
