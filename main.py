import models
import torchvision
from torchvision.transforms import ToTensor, Compose
import torch.nn as nn
from torch.utils.data import DataLoader
import itertools

from visualization import ActivationHandler, MeanActivationHandler, GradientHandler

import util
from util import _create_optimizer
import numpy as np
from collections import defaultdict, namedtuple

from argparse import ArgumentParser

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
        2 :class:`DataLoader`s are returned, obe for train and test set
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
        else:
            return labels.unique().numel()

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
    _handlers   :   defaultdict(list)
                    Dict of handlers for `activation`, `gradient`, `output`
    _supervisor :   Supervisor
                    Supervisor to use in the model
    _model  :   nn.Module
    _optimizer  : torch.optim.Optimizer
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
        self._dataloader = DataLoader(self._dataset, batch_size=self._batch_size, shuffle=True)
        self._data_iter = iter(self._dataloader)

        print(f'Number of classes: {self._num_classes}')
        print(f'Data shape: {self._shape}')
        self._handlers = defaultdict(list)
        self._supervisor = self._model = self._optimizer = None

    def supervise(self, *modules):
        '''Set the supervisor.

        Parameters
        ----------
        modules :   list(type)
                    Arbitrary number of :class:`nn.Module` types to supervise.
        '''
        from supervise import capture_modules
        self._supervisor = capture_modules(*modules)

    def optimize(self, **kwargs):
        '''Set the optimizer.

        Parameters
        ----------
        name    :   str
                    Name of the optimizer (must exist in :mod:`torch.optim`)

        All other kwargs are forwarded to the optimizer constructor
        '''
        name = kwargs.pop('name', 'Adam')
        self._optimizer = _create_optimizer(self._model, name, **kwargs)
        print(f'Using {self._optimizer.__class__.__name__} optimizer')

    def add_model(self, model_str):
        '''Set the model to train/test.
        .. warning::
            Currently, the function automatically calls :meth:`nn.Modue.cuda()` and hence a GPU is
            necessary.

        Parameters
        ----------
        model_str   :   str
                        Name of the model (must exist in :mod:`models`)
        '''
        Model = getattr(models, model_str)
        self._model = Model(self._shape, num_classes=self._num_classes, supervisor=self._supervisor)
        _initialize_model(self._model)
        self._model.cuda()

    def add_handlers(self, *handlers):
        '''Add handlers for processing activation, gradient, or output information. They will
        automatically be registered according to their type.

        Parameters
        ----------
        handlers    :   list(supervise.Handler)

        Raises
        ------
        ValueError
            In case there's a handler whose type is not :class:`ActivationHandler`,
            :class:`GradientHandler`.
        '''
        for handler in handlers:
            if isinstance(handler, ActivationHandler):
                self._handlers['activation'].append(handler)
                self._supervisor.register_activation_observer(handler)
            elif isinstance(handler, GradientHandler):
                self._handlers['gradient'].append(handler)
                self._supervisor.register_output_observer(handler)
            else:
                raise ValueError(f'Don\'t know what to do with {handler}')

    def train(self):
        '''Run through 1 batch in the training set. The iterator will wrap around and
        restart at the beginning.'''

        # to be safe, enable batch-norm, dropout, and the like. Could be changed externally, so
        # do this before each epoch
        self._model.train(True)
        try:
            X, Y = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._dataloader)
            X, Y = next(self._data_iter)
        data, labels = X.cuda(), Y.cuda()
        self._optimizer.zero_grad()
        output = self._model(data)
        loss = self._loss_function(output, labels)
        loss.backward()
        self._optimizer.step()

    def test(self, dataloader):
        '''Run through the test set once.

        Parameters
        ----------
        dataloader  :   torch.utils.DataLoader
                        Loader for the test data.
        '''
        self._model.train(False)
        for X, _labels in dataloader:
            self._model(X.cuda())


def _main(dataset_str, model_str, batch_size=512):

    # pep8: ignore=E221
    dataset_train, dataset_test = _load_dataset(dataset_str)
    N_train                     = len(dataset_train.dataset)
    N_test                      = len(dataset_test.dataset)
    batches_per_epoch           = int(N_train / batch_size + 0.5)
    test_batch_size             = 100
    test_loader                 = DataLoader(dataset_test.dataset, batch_size=test_batch_size)
    activation_handler          = MeanActivationHandler(
                                        int(N_train * 0.05)   # step every 5%
                                  )

    trainer = Trainer(dataset_train, batch_size=batch_size)
    trainer.supervise(nn.ReLU, nn.Linear)
    trainer.add_model(model_str)
    trainer.add_handlers(activation_handler)
    trainer.optimize(name='Adam')

    epochs = 1000
    for e in range(epochs):
        print(f'Starting epoch {e+1:5d} of {epochs:5d}')
        for batch_idx in range(batches_per_epoch):
            trainer.train()
            if batch_idx % 10 == 0:
                trainer.test(test_loader)


def main():
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

    args = parser.parse_args()
    _main(args.dataset, args.model)


if __name__ == '__main__':
    main()
