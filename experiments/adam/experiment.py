import torch
from torchvision.transforms import ToTensor, LinearTransformation
import numpy as np

##################
#  Ikkuna stuff  #
##################
from ikkuna.utils import load_dataset
from ikkuna.export import Exporter
from ikkuna.export.subscriber import (TestAccuracySubscriber,
                                      TrainAccuracySubscriber, BiasCorrectedMomentsSubscriber)
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


@ex.config
def cfg():
    base_lr    = 0.001
    optimizer  = 'SGD'
    batch_size = 128
    n_epochs   = 45
    loss       = 'CrossEntropyLoss'
    dataset    = 'CIFAR10'
    model      = 'AdamModel'
    schedule   = None
    identifier = EXPERIMENT_NAME


@ex.automain
def run(batch_size, loss, optimizer, base_lr, n_epochs, dataset, model):

    cifar_whitening_matrix = np.load('zca_matrix.npy')
    train_transforms = [LinearTransformation(cifar_whitening_matrix), ToTensor()]
    # load the dataset
    dataset_train_meta, dataset_test_meta = load_dataset(dataset,
                                                         train_transforms=train_transforms,
                                                         test_transforms=train_transforms)

    exporter = Exporter(depth=-1, module_filter=[torch.nn.Conv2d, torch.nn.Linear],)
    # instantiate model
    if model == 'AdamModel':
        model = AdamModel(exporter=exporter)
    else:
        from ikkuna.models import get_model
        model = get_model(model, dataset_train_meta.shape[1:],
                          num_classes=dataset_train_meta.num_classes, exporter=exporter)

    loss_fn = getattr(torch.nn, loss)()

    # set up the trainer
    trainer = Trainer(dataset_train_meta, batch_size=batch_size, loss=loss_fn, exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=optimizer, lr=base_lr)
    trainer.add_subscriber(BiasCorrectedMomentsSubscriber(0.9, 0.999, 1e-8))
    trainer.add_subscriber(TrainAccuracySubscriber())
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test_meta, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=batch_size))

    logged_metrics = ['loss',
                      'test_accuracy',
                      'train_accuracy',
                      'layer_gradients_norm2',
                      'weight_gradients_norm2',
                      'weight_gradients_variance',
                      'layer_gradients_variance',
                      'weight_gradients_variance',
                      'bias_corrected_gradient_mean',
                      'bias_corrected_gradient_var',
                      'lr_multiplier']

    trainer.add_subscriber(SacredLoggingSubscriber(ex, logged_metrics))

    # do n epochs of training
    batches_per_epoch = trainer.batches_per_epoch
    epochs            = n_epochs
    for i in range(epochs):
        for b in range(batches_per_epoch):
            trainer.train_batch()

    # we return a result so we can use it for filtering aborted experiments in mongodb
    return 0
