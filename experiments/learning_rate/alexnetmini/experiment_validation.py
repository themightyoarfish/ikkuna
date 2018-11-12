import torch
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger()

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
logger.setLevel(logging.WARNING)   # when run in sacred, log level seems to be higher
ex.logger = logger

from subscribers import RatioLRSubscriber, SacredLoggingSubscriber

# we need this so we can define a function for the learning rate like the exponential and identity
# fns. we will define the subscriber in the main function. I know, this hurts me too.
LR_SUBSCRIBER = None


schedules = [
    'identity_schedule_fn',
    'exponential_schedule_fn',
    'ratio_adaptive_schedule_fn'
]


def ratio_adaptive_schedule_fn(epoch):
    return LR_SUBSCRIBER(epoch)


def identity_schedule_fn(epoch):
    return 1


def oscillating_schedule_fn(epoch, period=5):
    # switch every 5 epochs between lr and lr/10
    return 1 if (epoch // period) % 2 == 0 else 0.1


def exponential_schedule_fn(epoch, gamma=0.98):
    return gamma ** epoch


@ex.config
def cfg():
    base_lr    = 0.2
    optimizer  = 'SGD'
    batch_size = 128
    n_epochs   = 100
    loss       = 'CrossEntropyLoss'
    schedule   = 'ratio_adaptive_schedule_fn'
    dataset    = 'CIFAR10',
    model      = 'AlexNetMini',


@ex.automain
def run(batch_size, loss, optimizer, base_lr, n_epochs, schedule, dataset, model):
    global LR_SUBSCRIBER
    LR_SUBSCRIBER = RatioLRSubscriber(base_lr)
    # load the dataset
    dataset_train_meta, dataset_test_meta = load_dataset(dataset)

    exporter = Exporter(depth=-1, module_filter=[torch.nn.Conv2d, torch.nn.Linear])
    # instantiate model
    from ikkuna import models
    try:
        if model.startswith('ResNet'):
            model_fn = getattr(models, model.lower())
            model = model_fn(exporter=exporter)
        else:
            Model = getattr(models, model)
            model = Model(dataset_train_meta.shape[1:], num_classes=dataset_train_meta.num_classes,
                          exporter=exporter)
    except AttributeError:
        raise ValueError(f'Unknown model {model}')

    # get loss and scheduling function since sacred can only log strings
    loss_fn     = getattr(torch.nn, loss)()
    schedule_fn = globals()[schedule]

    # set up the trainer
    trainer = Trainer(dataset_train_meta, batch_size=batch_size, loss=loss_fn, exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=optimizer, lr=base_lr)
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights']))
    trainer.add_subscriber(LR_SUBSCRIBER)
    trainer.add_subscriber(TrainAccuracySubscriber())
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test_meta, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=batch_size))

    trainer.add_subscriber(SacredLoggingSubscriber(ex, ['test_accuracy', 'learning_rate']))
    trainer.set_schedule(torch.optim.lr_scheduler.LambdaLR, schedule_fn)

    # do n epochs of training
    batches_per_epoch = trainer.batches_per_epoch
    epochs            = n_epochs
    for i in range(epochs):
        for b in range(batches_per_epoch):
            trainer.train_batch()

    # do testing batchwise to avoid memory errors
    n_batches = 0
    accuracy  = 0
    loader    = iter(DataLoader(dataset_test_meta.dataset, batch_size=batch_size,
                                shuffle=False, pin_memory=True))
    try:
        model.train(False)
        while True:
            X, labels   = next(loader)
            outputs     = model(X.cuda())
            predictions = outputs.argmax(1)
            n_correct   = (predictions.cpu() == labels).sum().item()
            accuracy   += n_correct / X.shape[0]
            n_batches  += 1
    except StopIteration:
        accuracy /= n_batches
        return accuracy
