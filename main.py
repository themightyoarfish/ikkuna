import models
import torchvision
from torchvision.transforms import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from supervise import EpochLossHandler, ActivationHandler, MeanActivationHandler

from run_model import train_epoch, _create_optimizer
import numpy as np
from collections import defaultdict

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

class Training:
    def __init__(self, dataset_str, **kwargs):
        '''Train a model on a dataset (only images). The model must be compatible with the image size.

        Parameters
        ----------
        batch_size  :   int
        loss   :   function
                            Defaults to nn.CrossEntropyLoss
        '''
        ############################################################################################
        #                                  Acquire parameters                                      #
        ############################################################################################
        self._dataset, self._num_classes, *self._shape = _load_dataset(dataset_str)
        self._batch_size    = kwargs.pop('batch_size', 1)
        self._loss_function = kwargs.pop('loss', nn.CrossEntropyLoss())
        self._dataloader    = DataLoader(self._dataset, batch_size=self._batch_size, shuffle=True)

        print(f'Number of classes: {self._num_classes}')
        print(f'Data shape: {self._shape}')
        self._handlers = defaultdict(list)
        self._supervisor = self._model = self._optimizer = None

    def supervise(self, *modules):
        from supervise import capture_modules
        self._supervisor = capture_modules(*modules)
        return self

    def optimizer(self, **kwargs):
        name = kwargs.pop('name', 'Adam')
        self._optimizer = _create_optimizer(self._model, name, **kwargs)
        print(f'Using {self._optimizer.__class__.__name__} optimizer')
        return self

    def model(self, model_str):
        Model = getattr(models, model_str)
        self._model = Model(self._shape, num_classes=self._num_classes, supervisor=self._supervisor)
        _initialize_model(self._model)
        self._model.cuda()
        return self

    def display(self, *handlers):
        for handler in handlers:
            if isinstance(handler, ActivationHandler):
                self._handlers['activation'].append(handler)
                self._supervisor.register_activation_observer(handler)
            else:
                self._handlers['output'].append(handler)
                self._supervisor.register_output_observer(handler)
        return self

    def train(self):
        # to be safe, enable batch-norm, dropout, and the like. Could be changed externally, so
        # do this before each epoch
        self._model.train(True)
        for h in self._handlers['activation'] + self._handlers['gradient']:
            h.on_epoch_started()
        train_epoch(self._model, self._dataloader, self._optimizer, self._loss_function)
        for h in self._handlers['activation'] + self._handlers['gradient']:
            h.on_epoch_finished()

    def test(self, dataloader):
        for h in self._handlers['output']:
            h.on_epoch_started()
        self._model.train(False)
        for X, _ in dataloader:
            self._model(X.cuda())
        for h in self._handlers['output']:
            h.on_epoch_finished()



def _main(dataset_str, model_str):

    dataset_test, *_rest = _load_dataset(dataset_str, train=False)
    test_batch_size = 100
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size)
    activation_handler = MeanActivationHandler()
    loss_handler = EpochLossHandler(test_loader, nn.CrossEntropyLoss())

    trainer = Training(dataset_str, batch_size=512).supervise(nn.ReLU, nn.Linear)
    trainer.model(model_str).display(activation_handler, loss_handler).optimizer(name='Adam')

    epochs = 1000
    for e in range(epochs):
        print(f'Starting epoch {e+1:5d} of {epochs:5d}')
        trainer.train()
        trainer.test(test_loader)

def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['AlexNetMini', 'DenseNet'], required=True)
    data_choices = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    parser.add_argument('-d', '--dataset', type=str, choices=data_choices, required=True)

    args = parser.parse_args()
    _main(args.dataset, args.model)

if __name__ == '__main__':
    main()
