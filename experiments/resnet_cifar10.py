'''
This experiment attempts to reproduces Microsofts original ResNet paper
'''
import sys
sys.path.append('.')
import random
import numpy as np
from tqdm import tqdm
import torch
from torchvision.transforms import Lambda, ToTensor

from ikkuna.export.subscriber import (TrainAccuracySubscriber, TestAccuracySubscriber,
                                      SpectralNormSubscriber, RatioSubscriber)
from ikkuna.utils import load_dataset
from ikkuna.models import resnet18
from train import Trainer
from schedulers import FunctionScheduler

train_config = {
    'base_lr':       0.1,
    'optimizer':     'SGD',
    'weight_decay':  0.0001,
    'momentum':      0.9,
    'batch_size':    128,
    'n_iters':       64_000,
    'loss':          torch.nn.CrossEntropyLoss(),
}


mean = np.load('./experiments/cifar10_mean.npy')


def whiten(img):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img, dtype=np.float32)
    img -= mean
    return img


def random_flip(img):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img, dtype=np.float32)

    # flip horizontally
    if bool(random.getrandbits(1)):
        img = np.fliplr(img)

    return img


def pad4(img):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img, dtype=np.float32)

    # pad with 4
    img = np.pad(img, [(4, 4), (4, 4), (0, 0)], mode='constant')
    return img


def random_crop_32x32_from_40x40(img):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img, dtype=np.float32)
    offset_x = random.randint(0, 7)
    offset_y = random.randint(0, 7)
    return img[offset_x:offset_x+32, offset_y:offset_y+32, :]


train_transforms = [Lambda(whiten), Lambda(random_flip), Lambda(pad4),
                    Lambda(random_crop_32x32_from_40x40), ToTensor()]
test_transforms = [Lambda(whiten), ToTensor()]


def schedule_fn(base_lrs, batch, step, epoch):
    if step == 32_000:
        print('Reducing LR by 10')
        ret = [lr / 10 for lr in base_lrs]
    elif step == 48_000:
        print('Reducing LR by 10 again')
        ret = [lr / 100 for lr in base_lrs]
    else:
        ret = base_lrs
    return ret


def initialize(m):
    # in PyTorch, all modules have a default initialization method, so you only need to override
    # specifics
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


dataset_train, dataset_test = load_dataset('CIFAR10', train_transforms=train_transforms,
                                           test_transforms=test_transforms)


def main():
    trainer = Trainer(dataset_train, batch_size=train_config['batch_size'],
                      loss=train_config['loss'])
    exporter = trainer.exporter
    model = resnet18(exporter=exporter, num_classes=dataset_train.num_classes)
    trainer.set_model(model)
    trainer.optimize(name=train_config['optimizer'], weight_decay=train_config['weight_decay'],
                     momentum=train_config['momentum'], lr=train_config['base_lr'])
    trainer.initialize(initialize)
    trainer.set_schedule(FunctionScheduler, schedule_fn)
    trainer.add_subscriber(TrainAccuracySubscriber())
    trainer.add_subscriber(TestAccuracySubscriber(dataset_test, trainer.model.forward,
                                                  frequency=2000,
                                                  batch_size=train_config['batch_size']))
    trainer.add_subscriber(SpectralNormSubscriber('weights'))
    trainer.add_subscriber(RatioSubscriber(['weight_updates', 'weights']))

    for i in tqdm(range(train_config['n_iters'])):
        trainer.train_batch()


if __name__ == '__main__':
    main()

# TODO: Compute mean per pixel and subtract
#       Make scheduler
#       Make train loop over 64k iterations with random sampling (should be less than 2 epochs?)
