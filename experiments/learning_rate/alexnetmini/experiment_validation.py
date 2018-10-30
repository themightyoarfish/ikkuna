import torch
from torch.utils.data import DataLoader

##################
#  Ikkuna stuff  #
##################
from ikkuna.models import AlexNetMini
from ikkuna.utils import load_dataset
from train import Trainer

##################
#  Sacred stuff  #
##################
EXPERIMENT_NAME = __file__[:-3]
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment(EXPERIMENT_NAME, interactive=True)
ex.observers.append(MongoObserver.create())


schedules = ['identity_schedule_fn', 'exponential_schedule_fn', 'oscillating_schedule_fn']


def identity_schedule_fn(epoch):
    return 1


def oscillating_schedule_fn(epoch, period=5):
    # switch every 5 epochs between lr and lr/10
    return 1 if (epoch // period) % 2 == 0 else 0.1


def exponential_schedule_fn(epoch, gamma=0.98):
    return gamma ** epoch


@ex.config
def cfg():
    base_lr    = 0.5
    optimizer  = 'SGD'
    batch_size = 128
    n_epochs   = 100
    loss       = 'CrossEntropyLoss'
    schedule   = 'oscillating_schedule_fn'


@ex.main
def run(batch_size, loss, optimizer, base_lr, n_epochs, schedule):
    # load the dataset
    dataset_train_meta, dataset_test_meta = load_dataset('CIFAR10')

    # instantiate model
    model = AlexNetMini(dataset_train_meta.shape[1:],
                        num_classes=dataset_train_meta.num_classes)

    # get loss and scheduling function since sacred can only log strings
    loss_fn     = getattr(torch.nn, loss)()
    schedule_fn = globals()[schedule]

    # set up the trainer. Will create its own exporter, but we don't use it
    trainer = Trainer(dataset_train_meta, batch_size=batch_size, loss=loss_fn)
    trainer.set_model(model)
    trainer.optimize(name=optimizer, lr=base_lr)
    trainer.set_schedule(torch.optim.lr_scheduler.LambdaLR, schedule_fn)

    # do n epochs of training
    batches_per_epoch = trainer.batches_per_epoch
    epochs = n_epochs
    for i in range(epochs):
        print(f'Epoch {i}')
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
