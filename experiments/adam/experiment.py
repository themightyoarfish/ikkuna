import torch
from torchvision.transforms import ToTensor, LinearTransformation
from torch.utils.data import TensorDataset
import numpy as np

##################
#  Ikkuna stuff  #
##################
from ikkuna.utils import load_dataset, get_model
from ikkuna.export import Exporter
from ikkuna.export.subscriber import (TestAccuracySubscriber, TrainAccuracySubscriber,
                                      VarianceSubscriber, MeanSubscriber, SpectralNormSubscriber,
                                      RatioSubscriber, NormSubscriber)
from ikkuna.export.subscriber.loss import LossSubscriber
from train import Trainer

##################
#  Sacred stuff  #
##################
EXPERIMENT_NAME = __file__[:-3]
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment(EXPERIMENT_NAME, interactive=True)
ex.observers.append(MongoObserver.create())

from experiments.subscribers import SacredLoggingSubscriber
from adam_model import AdamModel
from running_grad_moments import BiasCorrectedMomentsSubscriber


@ex.config
def cfg():
    base_lr    = 0.001
    optimizer  = 'Adam'
    batch_size = 128
    n_epochs   = 45
    loss       = 'CrossEntropyLoss'
    dataset    = 'CIFAR10'
    model      = 'AdamModel'
    schedule   = None
    identifier = EXPERIMENT_NAME


@ex.automain
def run(batch_size, loss, optimizer, base_lr, n_epochs, dataset, model):
    # load the dataset
    dataset_train_meta, dataset_test_meta = load_dataset(dataset)
    # whitening doesn't seem to work ???
    # if dataset == 'CIFAR10':
    #     whitened_cifar_train = np.load('whitened_cifar_train.npy').transpose([0, 2, 3, 1])
    #     whitened_cifar_test = np.load('whitened_cifar_test.npy').transpose([0, 2, 3, 1])
    #     dataset_train_meta.dataset.data = whitened_cifar_train
    #     dataset_test_meta.dataset.data = whitened_cifar_test

    exporter = Exporter(depth=-1, module_filter=[torch.nn.Conv2d, torch.nn.Linear],)
    # instantiate model
    if model == 'AdamModel':
        model = AdamModel(dataset_test_meta.shape[1:],
                          num_classes=dataset_train_meta.num_classes, exporter=exporter)
    else:
        model = get_model(model, dataset_train_meta.shape[1:],
                          num_classes=dataset_train_meta.num_classes, exporter=exporter)

    loss_fn = getattr(torch.nn, loss)()

    # set up the trainer
    trainer = Trainer(dataset_train_meta, batch_size=batch_size, loss=loss_fn,
                      exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=optimizer, lr=base_lr)
    trainer.add_subscriber(BiasCorrectedMomentsSubscriber(0.9, 0.999, 1e-8, backend=None))
    trainer.add_subscriber(LossSubscriber(backend=None))
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights'], backend=None))
    trainer.add_subscriber(NormSubscriber('weight_gradients', backend=None))
    trainer.add_subscriber(SpectralNormSubscriber('weights', backend=None))
    trainer.add_subscriber(VarianceSubscriber('weight_gradients', backend=None))
    trainer.add_subscriber(MeanSubscriber('weight_gradients', backend=None))
    trainer.add_subscriber(TrainAccuracySubscriber(backend=None))
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test_meta, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=batch_size, backend=None))

    logged_metrics = ['loss',
                      'test_accuracy',
                      'train_accuracy',
                      'weight_gradients_mean',
                      'weight_gradients_variance',
                      'weights_spectral_norm',
                      'weight_updates_weights_ratio',
                      'lr_multiplier',
                      'biased_grad_mean_estimate_mean',
                      'biased_grad_mean_estimate_var',
                      'biased_grad_var_estimate_mean',
                      'biased_grad_var_estimate_var',
                      'grad_mean_estimate_mean',
                      'grad_mean_estimate_var',
                      'grad_var_estimate_mean',
                      'grad_var_estimate_var',
                      'lr_multiplier_mean',
                      'lr_multiplier_var',
                      ]

    trainer.add_subscriber(SacredLoggingSubscriber(ex, logged_metrics))

    # do n epochs of training
    batches_per_epoch = trainer.batches_per_epoch
    epochs            = n_epochs
    for i in range(epochs):
        for b in range(batches_per_epoch):
            trainer.train_batch()

    # we return a result so we can use it for filtering aborted experiments in mongodb
    return 0
