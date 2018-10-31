import torch.nn as nn
from torch.utils.data import DataLoader
from ikkuna.utils import create_optimizer
from ikkuna.export import Exporter
from typing import NamedTuple


class DatasetMeta(NamedTuple):
    '''Class encapsulating a dataset and makes information about it more easily accessible.'''
    dataset: object
    num_classes: int
    shape: tuple

    @property
    def size(self):
        '''int: Number of examples in the dataset'''
        return self.shape[0]


class Trainer:
    '''Class to bundle all logic and parameters that go into training a model on some
    dataset.

    Attributes
    ----------
    _dataset :  torch.utils.data.Dataset
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
    _optimizer  : torch.optim.Optimizer
    _scheduler  :   torch.optim.lr_scheduler._LRScheduler
    '''

    def __init__(self, dataset_meta, **kwargs):
        '''Create a new Trainer. Handlers, model and optimizer are left uninitialised and must be
        set with :meth:`~train.Trainer.add_subscriber()`, :meth:`~train.Trainer.set_model()` and
        :meth:`~train.Trainer.optimize()` before calling :meth:`~train.Trainer.train_batch()`.

        Parameters
        ----------
        dataset_meta    :   train.DatasetMeta
                            Train data, obtained via :func:`ikkuna.utils.load_dataset()`. Currently,
                            only full batches are used; if the batch size does not evenly divide the
                            number of examples, the last batch is dropped.
        batch_size  :   int
        loss   :    function
                    Defaults to torch.nn.CrossEntropyLoss
        depth   :   int
                    Depth to which to traverse the module tree. Ignored if ``exporter`` keyword arg
                    is set
        '''
        ############################################################################################
        #                                  Acquire parameters                                      #
        ############################################################################################
        self._dataset, self._num_classes, self._shape = dataset_meta
        self._batch_size        = kwargs.pop('batch_size', 1)
        self._loss_function     = kwargs.pop('loss', nn.CrossEntropyLoss())
        self._dataloader        = DataLoader(self._dataset, batch_size=self._batch_size,
                                             pin_memory=True, shuffle=True, drop_last=True)
        self._data_iter         = iter(self._dataloader)
        N_train                 = self._shape[0]
        self._batches_per_epoch = N_train // self._batch_size
        self._batch_counter     = 0
        self._global_counter    = 0
        self._epoch             = 0
        self._scheduler         = None

        # we use these to peek one step ahead in the data iterator to know an epoch has ended
        # already in the epoch's final iteration, not at the beginning of the next one
        self._next_X, self._next_Y = next(self._data_iter)

        print(f'Number of classes: {self._num_classes}')
        print(f'Data shape: {self._shape}')
        self._exporter = kwargs.get('exporter', Exporter(kwargs.get('depth', -1)))
        self._exporter.set_loss(self._loss_function)

    @property
    def current_batch(self):
        '''int: 0-based batch index'''
        return self._batch_counter

    @property
    def batches_per_epoch(self):
        '''int: number of batches in an epoch (assuming only full batches, I think)'''
        return self._batches_per_epoch

    @property
    def model(self):
        '''torch.nn.Module: Model'''
        return self._model

    @property
    def exporter(self):
        '''ikkuna.export.Exporter: Exporter used during training'''
        return self._exporter

    @property
    def optimizer(self):
        '''torch.optim.Optimizer: Optimizer in use, if set'''
        return self._optimizer

    def add_subscriber(self, subscriber):
        '''Add a subscriber.

        Parameters
        ----------
        subscriber  :   ikkuna.export.subscriber.Subscriber

        '''
        self._exporter.message_bus.register_subscriber(subscriber)

    def optimize(self, name='Adam', **kwargs):
        '''Set the optimizer.

        Parameters
        ----------
        name    :   str
                    Name of the optimizer (must exist in :mod:`torch.optim`)
        **kwargs
            All other kwargs are forwarded to the optimizer constructor
        '''
        self._optimizer = create_optimizer(self._model, name, **kwargs)
        print(f'Using {self._optimizer.__class__.__name__} optimizer')

    def initialize(self, init):
        '''Run an initilization funnction on :attr:`Trainer.model`

        Parameters
        ----------
        init    :   function
        '''
        self._model.apply(init)

    def set_schedule(self, Scheduler, *args, **kwargs):
        '''Set a scheduler to anneal the learning rate.

        Parameters
        ----------
        Scheduler   :   type
                        Class of the Scheduler to use (e.g.
                        :class:`~torch.optim.lr_scheduler.LambdaLR`)
        *args   :   list
                    Passed to the scheduler constructor
        **kwargs    :   dict
                        Passed to the scheduler constructor
        '''
        if not self._optimizer:
            raise ValueError('You must set the optimizer before setting the schedule.')
        self._scheduler = Scheduler(self._optimizer, *args, **kwargs)

    def set_model(self, model_or_str):
        '''Set the model to train. This method will attempt to load from :mod:`ikkuna.models` if a
        string is passed.

        .. warning::
            Currently, the function automatically calls :meth:`torch.nn.Module.cuda()` and hence a
            GPU is necessary.

        Parameters
        ----------
        model_or_str    :   torch.nn.Module or str
                            Model or name of the model (must exist in :mod:`ikkuna.models`)
        '''
        if isinstance(model_or_str, str):
            from ikkuna import models
            try:
                if model_or_str.startswith('ResNet'):
                    model_fn = getattr(models, model_or_str.lower())
                    self._model = model_fn(exporter=self._exporter)
                else:
                    Model = getattr(models, model_or_str)
                    self._model = Model(self._shape[1:], num_classes=self._num_classes,
                                        exporter=self._exporter)
            except AttributeError:
                raise ValueError(f'Unknown model {model_or_str}')
        else:
            self._model = model_or_str

        self._model.cuda()
        if self._exporter._model is None:
            self._exporter.set_model(self._model)

    def train_batch(self):
        '''Run through 1 batch in the training set. The iterator will wrap around and
        restart at the beginning.'''

        # to be safe, enable batch-norm, dropout, and the like. Could be changed externally, so
        # do this before each epoch
        self._model.train(True)

        X, Y         = self._next_X, self._next_Y
        data, labels = X.cuda(async=True), Y.cuda(async=True)
        self._optimizer.zero_grad()
        output       = self._model(data)
        loss         = self._loss_function(output, labels)
        loss.backward()
        self._optimizer.step()

        try:
            self._next_X, self._next_Y = next(self._data_iter)
        except StopIteration:
            if self._scheduler:
                self._scheduler.step(self._epoch)
            self._exporter.epoch_finished()
            self._batch_counter        = 0
            self._epoch               += 1
            self._data_iter            = iter(self._dataloader)
            self._next_X, self._next_Y = next(self._data_iter)
        else:
            if self._scheduler:
                self._scheduler.step(self._epoch)

        self._batch_counter  += 1
        self._global_counter += 1
