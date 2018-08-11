import torch
import torch.nn as nn
import models
from torch.utils.data import DataLoader
from ikkuna.utils import create_optimizer, initialize_model
from ikkuna.export import Exporter
from collections import namedtuple


class DatasetMeta(namedtuple('DatasetMeta', ['dataset', 'num_classes', 'shape'])):
    @property
    def size(self):
        return self.shape[0]


class Trainer:
    '''Class to bundle all logic and parameters that go into training a model on some
    dataset.

    Attributes
    ----------
    _dataset :  Dataset
                The dataset used for training
    _num_classes    :   int
                        Number of target categories (inferred)
    _shape  :   list
                Shape of the input data (N, H, W, C)
    _batch_size :   int
                    Training batch size
    _loss_function  :   torch.nn._Loss
                        Loss function instance for training
    _dataloader :   torch.utils.data.DataLoader
                    loader for the training dataset
    _model  :   torch.nn.Module
    _optimizer  : torch.optim.Optimizer
    _exporter   :   ikkuna.export.Exporter
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
                    Defaults to torch.nn.CrossEntropyLoss
        '''
        ############################################################################################
        #                                  Acquire parameters                                      #
        ############################################################################################
        self._dataset, self._num_classes, self._shape = dataset_meta
        self._batch_size        = kwargs.pop('batch_size', 1)
        self._loss_function     = kwargs.pop('loss', nn.CrossEntropyLoss())
        sampler                 = torch.utils.data.sampler.RandomSampler(self._dataset)
        self._dataloader        = DataLoader(self._dataset, batch_size=self._batch_size,
                                             sampler=sampler, pin_memory=True)
        self._data_iter         = iter(self._dataloader)
        N_train                 = self._shape[0]
        self._batches_per_epoch = round(N_train / self._batch_size + 0.5)
        self._batch_counter     = 0

        # we use these to peek one step ahead in the data iterator to know an epoch has ended
        # already in the epoch's final iteration, not at the beginning of the next one
        self._next_X, self._next_Y = next(self._data_iter)

        print(f'Number of classes: {self._num_classes}')
        print(f'Data shape: {self._shape}')
        self._exporter = Exporter()

    @property
    def current_batch(self):
        return self._batch_counter

    @property
    def batches_per_epoch(self):
        return self._batches_per_epoch

    @property
    def model(self):
        return self._model

    @property
    def exporter(self):
        return self._exporter

    def add_subscriber(self, subscription):
        self._exporter.subscribe(subscription)

    def optimize(self, **kwargs):
        '''Set the optimizer.

        Parameters
        ----------
        name    :   str
                    Name of the optimizer (must exist in :mod:`torch.optim`)
        **kwargs
            All other kwargs are forwarded to the optimizer constructor
        '''
        name = kwargs.pop('name', 'Adam')
        self._optimizer = create_optimizer(self._model, name, **kwargs)
        print(f'Using {self._optimizer.__class__.__name__} optimizer')

    def add_model(self, model_str):
        '''Set the model to train.

        .. warning::
            Currently, the function automatically calls :meth:`torch.nn.Module.cuda()` and hence a
            GPU is necessary.

        Parameters
        ----------
        model_str   :   str
                        Name of the model (must exist in :mod:`models`)
        '''
        Model = getattr(models, model_str)
        self._model = Model(self._shape[1:], num_classes=self._num_classes, exporter=self._exporter)
        initialize_model(self._model)
        self._model.cuda()

    def train_batch(self):
        '''Run through 1 batch in the training set. The iterator will wrap around and
        restart at the beginning.'''

        # to be safe, enable batch-norm, dropout, and the like. Could be changed externally, so
        # do this before each epoch
        self._model.train(True)

        X, Y = self._next_X, self._next_Y
        data, labels = X.cuda(async=True), Y.cuda(async=True)
        self._optimizer.zero_grad()
        output = self._model(data)
        loss   = self._loss_function(output, labels)
        loss.backward()
        self._optimizer.step()

        try:
            self._next_X, self._next_Y = next(self._data_iter)
        except StopIteration:
            self._exporter.epoch_finished()
            self._batch_counter = 0
            self._data_iter = iter(self._dataloader)
            self._next_X, self._next_Y = next(self._data_iter)

        self._batch_counter += 1
