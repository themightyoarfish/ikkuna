'''
.. moduleauthor:: Rasmus Diederichsen <rasmus@peltarion.com>

This module contains functions and classes for simplifying the training of ANN classifiers. It
accepts the following arguments:

.. argparse::
   :filename: ../main.py
   :func: get_parser
   :prog: main.py
'''
####################
#  stdlib imports  #
####################
from argparse import ArgumentParser, ArgumentTypeError
import time
import random
import os

#######################
#  3rd party imports  #
#######################
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose

#######################
#  1st party imports  #
#######################
from train import Trainer, DatasetMeta
from ikkuna.export.subscriber import (RatioSubscriber, HistogramSubscriber, SpectralNormSubscriber,
                                      TestAccuracySubscriber, TrainAccuracySubscriber)

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = '0'


def _load_dataset(name):
    '''Retrieve a dataset and determine the number of classes. This estimate is
    obtained from the number of different values in the training labels.

    Parameters
    ----------
    name    :   str
                Dataset name in :mod:`torchvision.datasets`

    Returns
    -------
    tuple
        2 :class:`train.DatasetMeta` s are returned, one for train and one test set
    '''
    transforms = Compose([ToTensor()])
    try:
        dataset_cls   = getattr(torchvision.datasets, name)
        dataset_train = dataset_cls('/home/share/data',
                                    download=True,
                                    train=True,
                                    transform=transforms
                                    )
        dataset_test  = dataset_cls('/home/share/data',
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
        N, H, W = data.shape[:3]
        C = data.shape[-1] if len(data.shape) == 4 else 1
        return (N, H, W, C)

    meta_train = DatasetMeta(dataset=dataset_train, num_classes=num_classes(dataset_train),
                             shape=shape(dataset_train))
    meta_test  = DatasetMeta(dataset=dataset_test, num_classes=num_classes(dataset_test),
                             shape=shape(dataset_test))

    return meta_train, meta_test


def _main(dataset_str, model_str, batch_size, epochs, optimizer, **kwargs):
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

    trainer = Trainer(dataset_train, batch_size=batch_size)
    trainer.add_model(model_str)
    trainer.optimize(name=optimizer)

    subsample = kwargs['subsample']
    backend   = kwargs['visualisation']

    if kwargs['spectral_norm']:
        spectral_norm_subscriber = SpectralNormSubscriber('weights',
                                                          subsample=subsample,
                                                          backend=backend)
        trainer.add_subscriber(spectral_norm_subscriber)

    if kwargs['test_accuracy']:
        test_accuracy_subscriber = TestAccuracySubscriber(dataset_test, trainer.model.forward,
                                                          frequency=trainer.batches_per_epoch)
        trainer.add_subscriber(test_accuracy_subscriber)

    if kwargs['train_accuracy']:
        train_accuracy_subscriber = TrainAccuracySubscriber(subsample=subsample,
                                                            backend=backend)
        trainer.add_subscriber(train_accuracy_subscriber)

    if kwargs['ratio']:
        for kind1, kind2 in kwargs['ratio']:
            ratio_subscriber = RatioSubscriber([kind1, kind2],
                                               subsample=subsample,
                                               backend=backend)
            trainer.add_subscriber(ratio_subscriber)

    if kwargs['histogram']:
        for kind in kwargs['histogram']:
            histogram_subscriber = HistogramSubscriber([kind], backend=backend)
            trainer.add_subscriber(histogram_subscriber)

    batches_per_epoch = trainer.batches_per_epoch
    print(f'Batches per epoch: {batches_per_epoch}')

    cum_time  = 0
    n_batches = 0
    for e in range(epochs):
        for batch_idx in range(batches_per_epoch):

            t0 = time.time()
            trainer.train_batch()
            t1 = time.time()

            n_batches += 1
            cum_time += t1 - t0

            if kwargs['verbose']:
                print(f'\repoch {e:>5d}/{epochs-1:<5d} '
                      f'| batch {batch_idx:>5d}/{batches_per_epoch-1:<5d} '
                      f'| {1. / (cum_time / n_batches):<3.1f} b/s', end='')

            if batch_idx % 20 == 0:
                cum_time = 0
                n_batches = 0


def get_parser():
    '''Obtain a configured argument parser. This function is necessary for the sphinx argparse
    extension.

    Returns
    -------
    argparse.ArgumentParser
    '''

    def list_of_tuples(input_):
        '''argparse type for passing a list of tuples'''
        try:
            kind1, kind2 = input_.split(',')
            return (kind1, kind2)
        except:     # noqa
            raise ArgumentTypeError('Values must be passed as val1,val2 (without space)')

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['AlexNetMini', 'DenseNet'],
                        required=True, help='Model class to train')
    data_choices = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    parser.add_argument('-d', '--dataset', type=str, choices=data_choices, required=True,
                        help='Dataset to train on')
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-o', '--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('-a', '--ratio-average', type=int, default=10, help='Number of ratios to '
                        'average for stability (currently unused)')
    parser.add_argument('-s', '--subsample', type=int, default=1,
                        help='Number of batches to ignore between updates')
    # parser.add_argument('-y', '--ylims', nargs=2, type=int, default=None,
    #                     help='Y-axis limits for plots')
    parser.add_argument('-v', '--visualisation', type=str, choices=['tb', 'mpl'], default='tb',
                        help='Visualisation backend to use.')
    parser.add_argument('-V', '--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--spectral-norm', action='store_true',
                        help='Use spectral norm subscriber on weights')
    parser.add_argument('--histogram', nargs='*', type=str, default=None,
                        help='Use histogram subscriber(s)')
    parser.add_argument('--ratio', type=list_of_tuples, nargs='*', default=None,
                        help='Use ratio subscriber(s)')
    parser.add_argument('--test-accuracy', action='store_true',
                        help='Use test set accuracy subscriber')
    parser.add_argument('--train-accuracy', action='store_true',
                        help='Use train accuracy subscriber')
    return parser


def main():

    args = get_parser().parse_args()
    kwargs = vars(args)
    _main(kwargs.pop('dataset'), kwargs.pop('model'), kwargs.pop('batch_size'),
          kwargs.pop('epochs'), kwargs.pop('optimizer'), **vars(args))


if __name__ == '__main__':
    main()
