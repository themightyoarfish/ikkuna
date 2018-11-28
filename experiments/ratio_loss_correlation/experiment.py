import torch
from torch.utils.data import DataLoader

##################
#  Ikkuna stuff  #
##################
from ikkuna.utils import load_dataset
from ikkuna.export import Exporter
from ikkuna.export.subscriber import (RatioSubscriber, TestAccuracySubscriber,
                                      TrainAccuracySubscriber)
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


@ex.config
def cfg():
    base_lr    = 0.2
    optimizer  = 'SGD'
    batch_size = 128
    n_epochs   = 75
    loss       = 'CrossEntropyLoss'
    dataset    = 'CIFAR10'
    model      = 'AlexNetMini'
    schedule   = None


@ex.automain
def run(batch_size, loss, optimizer, base_lr, n_epochs, schedule, dataset, model):

    # load the dataset
    dataset_train_meta, dataset_test_meta = load_dataset(dataset)

    exporter = Exporter(depth=-1)
    # instantiate model
    from ikkuna.models import get_model
    model = get_model(model, dataset_train_meta.shape[1:],
                      num_classes=dataset_train_meta.num_classes, exporter=exporter)

    loss_fn = getattr(torch.nn, loss)()

    # set up the trainer
    trainer = Trainer(dataset_train_meta, batch_size=batch_size, loss=loss_fn, exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=optimizer, lr=base_lr)
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights']))
    trainer.add_subscriber(TrainAccuracySubscriber())
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test_meta, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=batch_size))

    logged_metrics = ['weight_updates_weights_ratio',
                      'loss',
                      'test_accuracy',
                      'train_accuracy']

    if schedule == 'ratio_loss_adaptive_schedule_fn':
        from experiments.subscribers import RatioLRSubscriber
        lr_sub = RatioLRSubscriber(base_lr)
        trainer.add_subscriber(lr_sub)
        trainer.set_schedule(torch.optim.lr_scheduler.LambdaLR, lr_sub)
        logged_metrics.append('learning_rate')

    trainer.add_subscriber(SacredLoggingSubscriber(ex, logged_metrics))

    # do n epochs of training
    batches_per_epoch = trainer.batches_per_epoch
    epochs            = n_epochs
    for i in range(epochs):
        for b in range(batches_per_epoch):
            trainer.train_batch()

    # we return a result so we can use it for filtering aborted experiments in mongodb
    return 0
