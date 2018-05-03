import models
import torchvision
from torchvision.transforms import *
import torch
import torch.nn as nn
from run_model import train
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


def _main(dataset_str, model_str):
    dataset, num_classes, H, W, C = _load_dataset(dataset_str)
    dataset_test, *_rest = _load_dataset(dataset_str, train=False)
    test_batch_size = 100
    print(f'Number of classes: {num_classes}')
    print(f'Data shape: {(H, W, C)}')

    from supervise import capture_modules, MeanActivationHandler, EpochLossHandler
    supervisor = capture_modules(nn.ReLU, nn.Linear)
    Model = getattr(models, model_str)
    model = Model((H, W, C), num_classes=num_classes, supervisor=supervisor)
    _initialize_model(model)
    model.cuda()

    activation_handler = MeanActivationHandler()
    loss_handler = EpochLossHandler(dataset_test, nn.CrossEntropyLoss(), batch_size=test_batch_size)
    supervisor.register_activation_observer(activation_handler)
    supervisor.register_output_observer(loss_handler)

    def evaluate_epoch(model):
        activation_handler.on_epoch_finished()

        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset_test, batch_size=test_batch_size)
        model.train(False)
        for X, _ in dataloader:
            model(X.cuda())
        loss_handler.on_epoch_finished()
        model.train(True)

    train(model, dataset, post_epoch_hook=evaluate_epoch, batch_size=1000, epochs=400)

def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['AlexNetMini', 'DenseNet'], required=True)
    data_choices = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    parser.add_argument('-d', '--dataset', type=str, choices=data_choices, required=True)

    args = parser.parse_args()
    _main(args.dataset, args.model)

if __name__ == '__main__':
    main()
