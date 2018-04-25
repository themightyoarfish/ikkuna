import torch.nn as nn

from torch.utils import DataSet
from torch.utils.data import DataLoader
from functools import partial
from torch.optim import Adam

def _get_optimizer(model, **kwargs):
    learning_rate = kwargs.get('learning_rate', 1e-4)
    optimizer = kwargs.get('optimizer', Adam([p for p in model.parameters() if p.requires_grad()],
                                             lr=learning_rate))
    return optimizer

def train(model: nn.Module, dataset: DataSet, **kwargs):
    batch_size = kwargs.get('batch_size', 1)
    dataloader = DataLoader(dataset, shuffle=True)
    loss_function = kwargs.get('loss', nn.CrossEntropyLoss())
    optimizer = _get_optimizer(model, **kwargs)

    for X, Y in dataloader:
        output = model(X)
        loss = loss_function(output, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
