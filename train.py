import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam

def _get_optimizer(model, **kwargs):
    learning_rate = kwargs.get('learning_rate', 1e-4)
    optimizer = kwargs.get('optimizer', Adam([p for p in model.parameters() if p.requires_grad],
                                             lr=learning_rate))
    return optimizer

def train(model: nn.Module, dataset: Dataset, **kwargs):
    model.train()
    batch_size = kwargs.get('batch_size', 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_function = kwargs.get('loss', nn.CrossEntropyLoss())
    optimizer = _get_optimizer(model, **kwargs)
    epochs = kwargs.get('epochs', 1)

    for e in range(epochs):
        for idx, (X, Y) in enumerate(dataloader):
            print(f'\rIteration {idx:10d} of {len(dataloader):10d}', end='')
            data, labels = X.cuda(), Y.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
        print(f'\nEpoch {e} loss: {loss.item()}')
