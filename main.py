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

#######################
#  3rd party imports  #
#######################
from tqdm import tqdm

#######################
#  1st party imports  #
#######################
from train import Trainer
from ikkuna.utils import load_dataset
from ikkuna.export.subscriber import (RatioSubscriber, HistogramSubscriber, SpectralNormSubscriber,
                                      TestAccuracySubscriber, TrainAccuracySubscriber,
                                      NormSubscriber)
from ikkuna.export import Exporter
from ikkuna.export.messages import MessageBus
import ikkuna.visualization


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

    dataset_train, dataset_test = load_dataset(dataset_str)

    # for some strange reason, python claims 'torch referenced before assignment' when importing at
    # the top. hahaaaaa
    import torch
    bus = MessageBus('main')
    trainer = Trainer(dataset_train, batch_size=batch_size,
                      exporter=Exporter(depth=kwargs['depth'],
                                        module_filter=[torch.nn.Conv2d, torch.nn.Linear],
                                        message_bus=bus))
    trainer.set_model(model_str)
    trainer.optimize(name=optimizer, lr=kwargs.get('learning_rate', 0.01))
    if 'exponential_decay' in kwargs:
        decay = kwargs['exponential_decay']
        if decay is not None:
            import torch
            trainer.set_schedule(torch.optim.lr_scheduler.ExponentialLR, decay)

    subsample = kwargs['subsample']
    backend   = kwargs['visualisation']

    if kwargs['spectral_norm']:
        spectral_norm_subscriber = SpectralNormSubscriber('weights',
                                                          subsample=subsample,
                                                          backend=backend)
        trainer.add_subscriber(spectral_norm_subscriber)

    if kwargs['test_accuracy']:
        test_accuracy_subscriber = TestAccuracySubscriber(dataset_test, trainer.model.forward,
                                                          frequency=trainer.batches_per_epoch,
                                                          batch_size=batch_size,
                                                          backend=backend)
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

    if kwargs['norm']:
        for kind in kwargs['norm']:
            norm_subscriber = NormSubscriber([kind], backend=backend)
            trainer.add_subscriber(norm_subscriber)

    batches_per_epoch = trainer.batches_per_epoch
    print(f'Batches per epoch: {batches_per_epoch}')

    epoch_range = range(epochs)
    batch_range = range(batches_per_epoch)
    if kwargs['verbose']:
        epoch_range = tqdm(epoch_range, desc='Epoch')
        batch_range = tqdm(batch_range, desc='Batch')

    for e in epoch_range:
        for batch_idx in batch_range:
            trainer.train_batch()


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
    parser.add_argument('-m', '--model', type=str, required=True, help='Model class to train')
    data_choices = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    parser.add_argument('-d', '--dataset', type=str, choices=data_choices, required=True,
                        help='Dataset to train on')
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-o', '--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-a', '--ratio-average', type=int, default=10, help='Number of ratios to '
                        'average for stability (currently unused)', metavar='N')
    parser.add_argument('-s', '--subsample', type=int, default=1,
                        help='Number of batches to ignore between updates')
    # parser.add_argument('-y', '--ylims', nargs=2, type=int, default=None,
    #                     help='Y-axis limits for plots')
    parser.add_argument('-v', '--visualisation', type=str, choices=['tb', 'mpl'], default='tb',
                        help='Visualisation backend to use.')
    parser.add_argument('-V', '--verbose', action='store_true', help='Print training progress')
    parser.add_argument('--spectral-norm', action='store_true',
                        help='Use spectral norm subscriber on weights')
    parser.add_argument('--histogram', nargs='*', type=str, default=None, metavar='TOPIC',
                        help='Use histogram subscriber(s)')
    parser.add_argument('--ratio', type=list_of_tuples, nargs='*', default=None, metavar='TOPIC',
                        help='Use ratio subscriber(s)')
    parser.add_argument('--test-accuracy', action='store_true',
                        help='Use test set accuracy subscriber')
    parser.add_argument('--train-accuracy', action='store_true',
                        help='Use train accuracy subscriber')
    parser.add_argument('--norm', nargs='*', type=str, default=None, metavar='TOPIC',
                        help='Use norm subscriber(s)')
    parser.add_argument('--depth', type=int, default=-1, help='Depth to which to add modules',
                        metavar='N')
    parser.add_argument('--exponential-decay', type=float, required=False,
                        help='Decay parameter for exponential decay', metavar='GAMMA')
    return parser


def main():

    args = get_parser().parse_args()
    kwargs = vars(args)
    ikkuna.visualization.TBBackend.info = str(kwargs)
    _main(kwargs.pop('dataset'), kwargs.pop('model'), kwargs.pop('batch_size'),
          kwargs.pop('epochs'), kwargs.pop('optimizer'), **vars(args))


if __name__ == '__main__':
    main()
