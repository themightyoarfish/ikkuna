'''
.. moduleauthor:: Rasmus Diederichsen <rasmus@peltarion.com>

.. module:: main

This module contains functions and classes for simplifying the training of ANN classifiers. It
accepts the following arguments:

.. argparse::
   :filename: ../main.py
   :func: get_parser
   :prog: main.py
'''
import numpy as np
from collections import namedtuple
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, Compose

from ikkuna import models
from ikkuna.utils import create_optimizer
from ikkuna.export import Exporter

DatasetMeta = namedtuple('DatasetMeta', ['dataset', 'num_classes', 'shape'])


def _load_dataset(name):
    '''Retrieve a dataset and determine the number of classes. This estimate is
    obtained from the number of different values in the training labels.

    Parameters
    ----------
    name    :   str
                Dataset name in :py:mod:`torchvision.datasets`

    Returns
    -------
    tuple
        2 :class:`DataLoader`s are returned, one for train and one test set
    '''
    transforms = Compose([ToTensor()])
    try:
        dataset_cls = getattr(torchvision.datasets, name)
        dataset_train = dataset_cls('/home/share/data',
                                    download=True,
                                    train=True,
                                    transform=transforms
                                    )
        dataset_test = dataset_cls('/home/share/data',
                                   download=True,
                                   train=False,
                                   transform=transforms
                                   )
    except AttributeError:
        raise NameError(f'Dataset {name} unknown.')

    def num_classes(dataset):
        if dataset.train:
            _, labels = dataset.train_data, dataset.train_labels  # noqa
        else:
            _, labels = dataset.test_data, dataset.test_labels  # noqa

        # infer number of classes from labels. will fail if not all classes occur in labels
        if isinstance(labels, np.ndarray):
            return np.unique(labels).size()
        elif isinstance(labels, list):
            return len(set(labels))
        elif isinstance(labels, torch.Tensor):
            return labels.unique().numel()
        else:
            raise ValueError(f'Unexpected label storage {labels.__class__.__name__}')

    def shape(dataset):
        if dataset.train:
            data, _ = dataset.train_data, dataset.train_labels  # noqa
        else:
            data, _ = dataset.test_data, dataset.test_labels  # noqa

        # if only three dimensions, assume [N, H, W], else [N, H, W, C]
        H, W = data.shape[1:3]
        C = data.shape[-1] if len(data.shape) == 4 else 1
        return (H, W, C)

    meta_train = DatasetMeta(dataset=dataset_train, num_classes=num_classes(dataset_train),
                             shape=shape(dataset_train))
    meta_test = DatasetMeta(dataset=dataset_test, num_classes=num_classes(dataset_test),
                            shape=shape(dataset_test))

    return meta_train, meta_test


def _initialize_model(module, bias_val=0.01):
    '''Perform weight initialization on `module`. This is somewhat hacky since
    it assumes the presence of `weight` and/or `bias` fields on the module. Will
    skip if not present.

    Parameters
    ----------
    module  :   torch.nn.Module
                The model
    bias_val    :   float
                    Constant for biases (should be small and positive)

    Raises
    ------
    ValueError
        If ``module`` is not one of the known models (currently :class:`ikkuna.models.AlexNetMini`
        and :class:`ikkuna.models.DenseNet`)
    '''
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


class Trainer:
    '''Class to bundle all logic and parameters that go into training and testing a model on some
    dataset.

    Attributes
    ----------
    _dataset :  Dataset
                The dataset used for training (can differ from the test set)
    _num_classes    :   int
                        Number of target categories (inferred)
    _shape  :   list
                Shape of the input data (H, W, C)
    _batch_size :   int
                    Training batch size
    _loss_function  :   nn._Loss
                        Loss function instance for training
    _dataloader :   torch.utils.data.DataLoader
                    loader for the training dataset
    _model  :   nn.Module
    _optimizer  : torch.optim.Optimizer
    _train_counter  :   int
                        Counter for forward propagations
    _exporter   :   Exporter
    '''

    def __init__(self, dataset_meta: DatasetMeta, **kwargs):
        '''Create a new Trainer. Handlers, model and optimizer are left uninitialised and must be
        set with :meth:`supervise()`, :meth:`add_model` and :meth:`optimize` before calling :meth:
        `train`.

        .. warning::
            The order of calls must be exactly the one above, as the model must be initialised with
            the supervisor and the optimizer requires the model.

        Relevant keyword args are:

        Parameters
        ----------
        dataset_meta    :   DatasetMeta
                            Train data, obtained via :func:`_load_dataset()`
        batch_size  :   int
        loss   :    function
                    Defaults to nn.CrossEntropyLoss
        '''
        ############################################################################################
        #                                  Acquire parameters                                      #
        ############################################################################################
        self._dataset, self._num_classes, self._shape = dataset_meta
        self._batch_size = kwargs.pop('batch_size', 1)
        self._loss_function = kwargs.pop('loss', nn.CrossEntropyLoss())
        sampler = torch.utils.data.sampler.RandomSampler(self._dataset)
        self._dataloader = DataLoader(self._dataset, batch_size=self._batch_size, sampler=sampler)
        self._data_iter = iter(self._dataloader)

        print(f'Number of classes: {self._num_classes}')
        print(f'Data shape: {self._shape}')
        self._train_counter = 0
        self._exporter = Exporter()

    def optimize(self, **kwargs):
        '''Set the optimizer.

        Parameters
        ----------
        name    :   str
                    Name of the optimizer (must exist in :mod:`torch.optim`)

        All other kwargs are forwarded to the optimizer constructor
        '''
        name = kwargs.pop('name', 'Adam')
        self._optimizer = create_optimizer(self._model, name, **kwargs)
        print(f'Using {self._optimizer.__class__.__name__} optimizer')

    def add_model(self, model_str):
        '''Set the model to train/test.

        .. warning::
            Currently, the function automatically calls :meth:`nn.Module.cuda()` and hence a GPU is
            necessary.

        Parameters
        ----------
        model_str   :   str
                        Name of the model (must exist in :mod:`models`)
        '''
        Model = getattr(models, model_str)
        self._model = Model(self._shape, num_classes=self._num_classes, exporter=self._exporter)
        _initialize_model(self._model)
        self._model.cuda()

    def train(self):
        '''Run through 1 batch in the training set. The iterator will wrap around and
        restart at the beginning.'''

        # to be safe, enable batch-norm, dropout, and the like. Could be changed externally, so
        # do this before each epoch
        self._model.train(True)

        try:
            X, Y = next(self._data_iter)
        except StopIteration:
            self._exporter.epoch_finished()
            self._train_counter = 0
            self._data_iter     = iter(self._dataloader)
            X, Y                = next(self._data_iter)

        data, labels = X.cuda(), Y.cuda()
        self._optimizer.zero_grad()
        output = self._model(data)
        loss = self._loss_function(output, labels)
        loss.backward()
        self._optimizer.step()
        self._train_counter += 1
        self._exporter.step()

    def test(self, dataset):
        '''Run through the test set once.

        Parameters
        ----------
        dataset  :   DatasetMeta
        '''
        self._model.train(False)
        test_loader = DataLoader(dataset.dataset, batch_size=self._batch_size, shuffle=True)

        num_correct = 0
        n = 0
        for X, _labels in test_loader:
            n += X.shape[0]
            predictions = self._model(X.cuda()).argmax(1)
            num_correct += (predictions.cpu() == _labels).sum()
        return num_correct.item() / n


def _main(dataset_str, model_str, batch_size, epochs, optimizer):
    '''Run the training procedure.

    Parameters
    ----------
    dataset_str :   str
                    Name of the dataset to use
    model_str   :   str
                    Unqualified name of the model class to use
    batch_size  :   int
    epochs      :   int
    optimizer   :   str
                    Name of the optimizer to use
    '''

    dataset_train, dataset_test = _load_dataset(dataset_str)
    N_train                     = len(dataset_train.dataset)
    batches_per_epoch           = int(N_train / batch_size + 0.5)
    print(f'Data size: {N_train}')
    print(f'Batches per epoch: {batches_per_epoch}')

    trainer = Trainer(dataset_train, batch_size=batch_size)
    trainer.add_model(model_str)
    trainer.optimize(name=optimizer)

    for e in range(epochs):
        print(f'Starting epoch {e:5d} of {epochs:5d}')
        for batch_idx in range(batches_per_epoch):
            trainer.train()
        accuracy = trainer.test(dataset_test)
        print(f'Test accuracy: {accuracy}')


def get_parser():
    '''Obtain a configured argument parser. This function is necessary for the sphinx argparse
    extension.

    Returns
    -------
    argparse.ArgumentParser
    '''
    parser = ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        choices=[
            'AlexNetMini',
            'DenseNet'],
        required=True)
    data_choices = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    parser.add_argument('-d', '--dataset', type=str, choices=data_choices, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-o', '--optimizer', type=str, default='Adam')
    return parser


def main():

    args = get_parser().parse_args()
    _main(args.dataset, args.model, args.batch_size, args.epochs, args.optimizer)


if __name__ == '__main__':
    main()
