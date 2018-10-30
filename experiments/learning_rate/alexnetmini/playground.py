import sys
sys.path.append('.')
from tqdm import tqdm
import torch

from ikkuna.utils import load_dataset, seed_everything
seed_everything()

from ikkuna.export.subscriber import (TrainAccuracySubscriber, TestAccuracySubscriber,
                                      SpectralNormSubscriber, RatioSubscriber, VarianceSubscriber,
                                      NormSubscriber)
from ikkuna.export import Exporter
from ikkuna.models import AlexNetMini
import ikkuna.visualization
ikkuna.visualization.configure_prefix(__file__[:-3])
from train import Trainer


def identity_schedule_fn(epoch):
    return 1


def oscillating_schedule_fn(epoch, period=5):
    # switch every 5 epochs between lr and lr/10
    return 1 if (epoch // period) % 2 == 0 else 0.1


def exponential_schedule_fn(epoch, gamma=0.98):
    return gamma ** epoch


train_config = {
    'base_lr':    0.1,
    'optimizer':  'SGD',
    'batch_size': 1024,
    'n_epochs':   20,
    'loss':       torch.nn.CrossEntropyLoss(),
    'schedule':   exponential_schedule_fn
}


def main():
    dataset_train, dataset_test = load_dataset('MNIST')
    ikkuna.visualization.set_run_info('\n'.join(f'{k}: {v}' for k, v in train_config.items()))

    ikkuna.visualization.TBBackend.info = str(train_config)
    exporter = Exporter(depth=-1, module_filter=[torch.nn.Conv2d, torch.nn.Linear])
    trainer = Trainer(dataset_train, batch_size=train_config['batch_size'],
                      loss=train_config['loss'], exporter=exporter)
    model = AlexNetMini(dataset_train.shape[1:], num_classes=dataset_train.num_classes,
                        exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=train_config['optimizer'], lr=train_config['base_lr'])

    trainer.set_schedule(torch.optim.lr_scheduler.LambdaLR, train_config['schedule'])
    trainer.add_subscriber(TrainAccuracySubscriber())
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=train_config['batch_size']))
    # trainer.add_subscriber(SpectralNormSubscriber('weights'))
    # trainer.add_subscriber(SpectralNormSubscriber('weight_gradients'))
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights']))
    trainer.add_subscriber(VarianceSubscriber(['weight_updates']))
    trainer.add_subscriber(NormSubscriber(['weights']))
    trainer.add_subscriber(NormSubscriber(['weight_gradients']))

    batches_per_epoch = trainer.batches_per_epoch
    epochs = train_config['n_epochs']
    for i in tqdm(range(epochs), desc='Epoch'):
        for b in tqdm(range(batches_per_epoch), desc='Batch'):
            trainer.train_batch()


if __name__ == '__main__':
    main()
