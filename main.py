import models
import torchvision
from torchvision.transforms import *
import torch
import torch.nn as nn
from run_model import train, test
import numpy as np

from argparse import ArgumentParser

def _load_dataset(name, train=True):
    '''Retrieve a dataset and determine the number of classes. This estimate is
    obtained from the number of different values in the training labels.

    Parameters
    ----------
    name    :   str
                Dataset name in :py:mod:`torchvision.datasets`
    train   :   bool
                If `False` load test data instead of training data.
    '''
    try:
        dataset_cls = getattr(torchvision.datasets, name)
        dataset = dataset_cls('/home/share/data',
                              download=True,
                              train=train,
                              transform=transforms.Compose([ToTensor()])
                              )
    except AttributeError:
        raise NameError(f'Dataset {name} unknown.')

    if train:
        data, labels = dataset.train_data, dataset.train_labels
    else:
        data, labels = dataset.test_data, dataset.test_labels

    # turn labels into numpy array
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    num_classes = len(np.unique(np.array(labels)))

    # if only three dimensions, assume [N, H, W], else [N, H, W, C]
    H, W = data.shape[1:3]
    C = data.shape[-1] if len(data.shape) == 4 else 1

    return dataset, num_classes, H, W, C

def _initialize_model(module, bias_val=0.01):
    '''Perform weight initialization on `module`. This is somewhat hacky since
    it assumes the presence of `weight` and/or `bias` fields on the module. Will
    skip if not present.'''
    if isinstance(module, models.AlexNetMini):
        for m in module.modules():
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias'):
                module.bias.data.fill_(bias_val)
    elif isinstance(module, models.DenseNet):
        # Official init from torch repo.
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    else:
        raise ValueError(f'Don\'t know how to initialize {module.__class__.__name__}')



def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['AlexNetMini', 'DenseNet'], required=True)
    data_choices = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    parser.add_argument('-d', '--dataset', type=str, choices=data_choices, required=True)

    args = parser.parse_args()
    dataset, num_classes, H, W, C = _load_dataset(args.dataset)
    print(f'Number of classes: {num_classes}')
    print(f'Data shape: {(H, W, C)}')
    model = getattr(models, args.model)((H, W, C), num_classes=num_classes)
    _initialize_model(model)
    model.cuda()

    hook = lambda model: test(model, _load_dataset(args.dataset, train=False)[0])
    train(model, dataset, post_epoch_hook=hook, batch_size=512, epochs=10)

if __name__ == '__main__':
    main()
