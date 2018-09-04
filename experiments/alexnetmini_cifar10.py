import sys
sys.path.append('.')
from tqdm import tqdm
import torch

from ikkuna.export.subscriber import (TrainAccuracySubscriber, TestAccuracySubscriber,
                                      SpectralNormSubscriber, RatioSubscriber)
from ikkuna.export import Exporter
from ikkuna.utils import load_dataset, seed_everything
from ikkuna.models import AlexNetMini
from ikkuna.visualization import configure_prefix
configure_prefix('alexnet')
from train import Trainer
from schedulers import FunctionScheduler
seed_everything()


def identity_schedule_fn(base_lrs, batch, step, epoch):
    return base_lrs


def oscillating_schedule_fn(base_lrs, batch, step, epoch):
    factor = 1 if epoch % 2 == 0 else 10
    return [base_lr * factor for base_lr in base_lrs]


def good_schedule_fn(base_lrs, batch, step, epoch):
    return [base_lr * (0.96 ** epoch) for base_lr in base_lrs]


train_config = {
    'base_lr':      0.2,
    'optimizer':    'SGD',
    'batch_size':   1024,
    'n_epochs':     100,
    'loss':         torch.nn.CrossEntropyLoss(),
    'schedule':     identity_schedule_fn,
}

if __name__ == '__main__':
    dataset_train, dataset_test = load_dataset('CIFAR10')


def main():
    exporter = Exporter(depth=-1, module_filter=[torch.nn.Conv2d, torch.nn.Linear])
    trainer = Trainer(dataset_train, batch_size=train_config['batch_size'],
                      loss=train_config['loss'], exporter=exporter)
    model = AlexNetMini(dataset_train.shape[1:], num_classes=100, exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=train_config['optimizer'], lr=train_config['base_lr'])
    trainer.set_schedule(FunctionScheduler, train_config['schedule'])
    trainer.add_subscriber(TrainAccuracySubscriber())
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=train_config['batch_size']))
    trainer.add_subscriber(SpectralNormSubscriber('weights'))
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights']))

    batches_per_epoch = trainer.batches_per_epoch
    epochs = train_config['n_epochs']
    for i in tqdm(range(epochs), desc='Epoch'):
        for b in tqdm(range(batches_per_epoch), desc='Batch'):
            trainer.train_batch()


if __name__ == '__main__':
    main()
