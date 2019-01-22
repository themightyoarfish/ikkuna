import torch
from torchvision.transforms import ToTensor

##################
#  Ikkuna stuff  #
##################
from ikkuna.utils import load_dataset, get_model
from ikkuna.export import Exporter
from ikkuna.export.subscriber import (TestAccuracySubscriber, TrainAccuracySubscriber,
                                      RatioSubscriber, NormSubscriber, SVCCASubscriber)
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
    base_lr    = 0.1
    optimizer  = 'SGD'
    batch_size = 512
    n_epochs   = 30
    loss       = 'CrossEntropyLoss'
    dataset    = 'CIFAR10'
    model      = 'VGG'
    freeze_at  = 0.99
    schedule   = None
    identifier = EXPERIMENT_NAME


@ex.automain
def run(batch_size, loss, optimizer, base_lr, n_epochs, dataset, model):
    # load the dataset
    transforms = [ToTensor()]
    dataset_train_meta, dataset_test_meta = load_dataset(dataset, train_transforms=transforms,
                                                         test_transforms=transforms)
    exporter = Exporter(depth=-1, module_filter=[torch.nn.Conv2d, torch.nn.Linear],)
    # instantiate model
    model = get_model(model, dataset_train_meta.shape[1:],
                      num_classes=dataset_train_meta.num_classes, exporter=exporter)

    loss_fn = getattr(torch.nn, loss)()

    backend = None
    # set up the trainer
    trainer = Trainer(dataset_train_meta, batch_size=batch_size, loss=loss_fn,
                      exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=optimizer, lr=base_lr)
    trainer.add_subscriber(SVCCASubscriber(dataset_test_meta, 500, trainer.model.forward,
                                           subsample=trainer.batches_per_epoch, backend=backend))
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights'], backend=backend))
    trainer.add_subscriber(NormSubscriber('weight_gradients', backend=backend))
    trainer.add_subscriber(TrainAccuracySubscriber(backend=backend))
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test_meta, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=batch_size, backend=backend))

    logged_metrics = ['test_accuracy',
                      'train_accuracy',
                      'weight_gradients_norm2',
                      'weight_updates_weights_ratio',
                      'self_similarity',
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
