import sys
sys.path.append('.')
from tqdm import tqdm
import torch

from ikkuna.utils import load_dataset

from ikkuna.export.subscriber import (TrainAccuracySubscriber, TestAccuracySubscriber,
                                      SpectralNormSubscriber, RatioSubscriber, VarianceSubscriber,
                                      NormSubscriber)
from ikkuna.export import Exporter
from ikkuna.models import AlexNetMini
import ikkuna.visualization
ikkuna.visualization.configure_prefix(__file__[:-3])
from train import Trainer

from subscribers import RatioLRSubscriber


train_config = {
    'base_lr':    0.2,
    'optimizer':  'SGD',
    'batch_size': 128,
    'n_epochs':   150,
    'loss':       torch.nn.CrossEntropyLoss(),
}


def main():
    dataset_train, dataset_test = load_dataset('CIFAR10')
    ikkuna.visualization.set_run_info('\n'.join(f'{k}: {v}' for k, v in train_config.items()))

    exporter = Exporter(depth=-1)
    trainer = Trainer(dataset_train, batch_size=train_config['batch_size'],
                      loss=train_config['loss'], exporter=exporter)
    model = AlexNetMini(dataset_train.shape[1:], num_classes=dataset_train.num_classes,
                        exporter=exporter)
    trainer.set_model(model)
    trainer.optimize(name=train_config['optimizer'], lr=train_config['base_lr'])

    # add all the ordinary subscribers
    trainer.add_subscriber(TrainAccuracySubscriber())
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test, trainer.model.forward,
                                                  frequency=trainer.batches_per_epoch,
                                                  batch_size=train_config['batch_size']))
    trainer.add_subscriber(SpectralNormSubscriber('weights'))
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights']))
    trainer.add_subscriber(VarianceSubscriber('activations'))
    trainer.add_subscriber(NormSubscriber('weights'))
    trainer.add_subscriber(NormSubscriber('activations'))
    trainer.add_subscriber(NormSubscriber('layer_gradients'))
    lr_sub = RatioLRSubscriber(train_config['base_lr'])
    trainer.add_subscriber(lr_sub)
    trainer.set_schedule(torch.optim.lr_scheduler.LambdaLR, lr_sub)

    batches_per_epoch = trainer.batches_per_epoch
    epochs = train_config['n_epochs']
    for i in tqdm(range(epochs), desc='Epoch'):
        for b in tqdm(range(batches_per_epoch), desc='Batch'):
            trainer.train_batch()


if __name__ == '__main__':
    main()
