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
from argparse import ArgumentParser

import torch
import torchvision
from torchvision.transforms import ToTensor, Compose

from train import Trainer, DatasetMeta


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
        H, W = data.shape[1:3]
        C = data.shape[-1] if len(data.shape) == 4 else 1
        return (H, W, C)

    meta_train = DatasetMeta(dataset=dataset_train, num_classes=num_classes(dataset_train),
                             shape=shape(dataset_train))
    meta_test  = DatasetMeta(dataset=dataset_test, num_classes=num_classes(dataset_test),
                             shape=shape(dataset_test))

    return meta_train, meta_test


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

    import time
    cum_time = 0
    n_batches = 0
    for e in range(epochs):
        for batch_idx in range(batches_per_epoch):

            t0 = time.time()
            trainer.train()
            t1 = time.time()

            n_batches += 1
            cum_time += t1-t0

            if batch_idx % 10 == 0:
                print(f'\repoch {e+1:>5d}/{epochs:<5d} '
                      f'| batch {batch_idx+1:>5d}/{batches_per_epoch:<5d} '
                      f'| {1. / (cum_time / n_batches):<3.1f} b/s', end='')

        accuracy = trainer.test(dataset_test)
        print('')
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
