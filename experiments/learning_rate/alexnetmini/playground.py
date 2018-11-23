import sys

import torch
from tqdm import tqdm

import ikkuna.visualization
from ikkuna.export import Exporter
from ikkuna.export.subscriber import (ConditionNumberSubscriber,
                                      NormSubscriber, RatioSubscriber,
                                      SpectralNormSubscriber,
                                      TestAccuracySubscriber,
                                      TrainAccuracySubscriber,
                                      VarianceSubscriber)
from ikkuna.models import AlexNetMini
from ikkuna.utils import load_dataset
from experiments.subscribers import RatioLRSubscriber
from train import Trainer

sys.path.append('.')


ikkuna.visualization.configure_prefix(__file__[:-3])


train_config = {
    'base_lr': 0.01,
    'max_lr':  0.2,
    'optimizer': 'SGD',
    'batch_size': 128,
    'n_epochs': 100,
    'loss': 'CrossEntropyLoss',
    'schedule': 'disturbed_cyclic_schedule_fn',
    'dataset': 'CIFAR10',
    'model': 'AlexNetMini',
}


def disturbed_cyclic_schedule_fn(epoch, half_cycle_len=2):
    if 20 <= epoch <= 30:
        return 1
    else:
        return cyclic_schedule_fn(epoch, half_cycle_len)


def cyclic_schedule_fn(epoch, half_cycle_len=2):
    import math
    cycle_index = math.floor(1 + epoch / (2 * half_cycle_len))
    x           = abs(epoch / half_cycle_len - 2 * cycle_index + 1)
    base_lr     = train_config['base_lr']
    max_lr      = train_config['max_lr']
    factor      = 1 + max(0, 1-x) * max_lr / base_lr - max(0, 1-x)
    return factor


def partial_exponential_schedule_fn(epoch, gamma=0.98):
    return gamma ** epoch if epoch < 50 else 1


def step_schedule_fn(epoch, factor=2, total_epochs=100):
    phase_len = total_epochs // 4
    phase_index = int(epoch // phase_len)
    multiplier = (1/factor)**phase_index
    return multiplier


def main():
    backend = 'tb'
    dataset_train, dataset_test = load_dataset(train_config['dataset'])
    ikkuna.visualization.set_run_info('\n'.join(f'{k}: {v}' for k, v in train_config.items()))

    exporter = Exporter(depth=-1)
    loss_fn  = getattr(torch.nn, train_config['loss'])()
    trainer  = Trainer(dataset_train, batch_size=train_config['batch_size'],
                       loss=loss_fn, exporter=exporter)
    model=AlexNetMini(dataset_train.shape[1:], num_classes=dataset_train.num_classes,
                      exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=train_config['optimizer'], lr=train_config['base_lr'])
    schedule_fn = globals()[train_config['schedule']]
    trainer.set_schedule(torch.optim.lr_scheduler.LambdaLR, schedule_fn)

    # add all the ordinary subscribers
    trainer.add_subscriber(TrainAccuracySubscriber(backend=backend))
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=train_config['batch_size'],
                                                  backend=backend))
    trainer.add_subscriber(SpectralNormSubscriber('weights', backend=backend))
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights'], backend=backend))
    trainer.add_subscriber(VarianceSubscriber('activations', backend=backend))
    trainer.add_subscriber(NormSubscriber('weights', backend=backend))
    trainer.add_subscriber(NormSubscriber('layer_gradients', backend=backend))
    trainer.add_subscriber(NormSubscriber('weight_gradients', backend=backend))

    batches_per_epoch = trainer.batches_per_epoch
    epochs = train_config['n_epochs']

    for i in tqdm(range(epochs), desc='Epoch'):
        for b in tqdm(range(batches_per_epoch), desc='Batch'):
            trainer.train_batch()


if __name__ == '__main__':
    main()
