import torch
import torch.nn as nn
import models
from torch.utils.data import DataLoader
from ikkuna.utils import create_optimizer, initialize_model
from ikkuna.export import Exporter
from collections import namedtuple


DatasetMeta = namedtuple('DatasetMeta', ['dataset', 'num_classes', 'shape'])


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
        self._batch_size    = kwargs.pop('batch_size', 1)
        self._loss_function = kwargs.pop('loss', nn.CrossEntropyLoss())
        sampler             = torch.utils.data.sampler.RandomSampler(self._dataset)
        self._dataloader    = DataLoader(self._dataset, batch_size=self._batch_size,
                                         sampler=sampler)
        self._data_iter     = iter(self._dataloader)

        # we use these to peek one step ahead in the data iterator to know an epoch has ended
        # already in the epoch's final iteration, not at the beginning of the next one
        self._next_X, self._next_Y = next(self._data_iter)

        print(f'Number of classes: {self._num_classes}')
        print(f'Data shape: {self._shape}')
        self._exporter      = Exporter()
        from ikkuna.export.subscriber import RatioSubscriber, SynchronizedSubscription
        subscriber = RatioSubscriber(average=5)
        subscription = SynchronizedSubscription(subscriber, ['weights', 'weight_updates'])
        self._exporter.subscribe(subscription)

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
        initialize_model(self._model)
        self._model.cuda()

    def train(self):
        '''Run through 1 batch in the training set. The iterator will wrap around and
        restart at the beginning.'''

        # to be safe, enable batch-norm, dropout, and the like. Could be changed externally, so
        # do this before each epoch
        self._model.train(True)

        X, Y = self._next_X, self._next_Y

        data, labels = X.cuda(), Y.cuda()
        self._optimizer.zero_grad()
        output = self._model(data)
        loss   = self._loss_function(output, labels)
        loss.backward()
        self._optimizer.step()

        try:
            self._next_X, self._next_Y = next(self._data_iter)
        except StopIteration:
            self._exporter.epoch_finished()
            self._data_iter            = iter(self._dataloader)
            self._next_X, self._next_Y = next(self._data_iter)

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
            n           += X.shape[0]
            predictions  = self._model(X.cuda()).argmax(1)
            num_correct += (predictions.cpu() == _labels).sum()
        return num_correct.item() / n
