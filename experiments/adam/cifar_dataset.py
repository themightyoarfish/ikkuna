import numpy as np
from torch.utils.data import Dataset
import os.path as path
import torch


class WhitenedCIFAR10(Dataset):

    def __init__(self, root='/home/share/data/whitened-cifar/', train=True, transform=None,
                 target_transform=None, **kwargs):

        # for unknown reasons, labels must be < num_classes, so we need to start at 0. otherwise the
        # cuda NLLoss function fails. dunno why
        if train:
            self.data = torch.tensor(
                np.load('/home/share/data/whitened-cifar/whitened_cifar_train_data.npy'),
                dtype=torch.float32)
            self.targets = torch.tensor(
                np.load('/home/share/data/whitened-cifar/whitened_cifar_train_labels.npy'),
                dtype=torch.long
            ) - 1
        else:
            self.data = torch.tensor(
                np.load('/home/share/data/whitened-cifar/whitened_cifar_test_data.npy'),
                dtype=torch.float32
            )
            self.targets = torch.tensor(
                np.load('/home/share/data/whitened-cifar/whitened_cifar_test_labels.npy'),
                dtype=torch.long
            ) - 1

        self.transform = transform
        self.target_transform = transform

    def __getitem__(self, index):
        img, target = self.data[index, ...], self.targets[index]

        try:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
        except:
            raise ValueError('This is a tensor dataset')

        return img, target

    def __len__(self):
        return self.data.size(0)
