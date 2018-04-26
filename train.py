import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam

def _available_optimizers():
    '''List names of all available torch optimizers form :py:mod:`torch.optim`.

    Returns
    -------
    list(str)
    '''
    available_optimizers = set(getattr(torch.optim, name) for name in dir(torch.optim) if
            not name.startswith('__'))
    available_optimizers = filter(lambda o: type(o) == type, available_optimizers)
    available_optimizers = map(lambda o: o.__name__, available_optimizers)
    return list(available_optimizers)

def _create_optimizer(model, name, **kwargs):
    '''Create an optimizer for `model`s parameters. Will disregard all params
    witwith `requires_grad == False`.

    Parameters
    ----------
    model   :   nn.Module
    name    :   str
                Name of the optimizer

    **kwargs    :   dict
                    All arguments which should be passed to the optimizer.

    Raises
    ------
    ValueError
        If superflous `kwargs` are passed.

    '''

    if name not in _available_optimizers():
        raise ValueError(f'Unknown optimizer {name}')

    params = [p for p in model.parameters() if p.requires_grad]
    return getattr(torch.optim, name)(params, **kwargs)


def train(model: nn.Module, dataset: Dataset, **kwargs):
    '''Train a model on a dataset (only images). The model must be compatible with the image size.

    Parameters
    ----------
    model   :   nn.Module
    dataset :   Dataset
    batch_size  :   int
                    Defaults to 1
    epochs  :   int
                Defaults to 1
    loss_function   :   function
                        Defaults to nn.CrossEntropyLoss
    optimizer   :   torch.optim.Optimizer
                    Defaults to Adam with 1e-4 learning rate
    '''
    model.train()
    batch_size    = kwargs.pop('batch_size', 1)
    epochs        = kwargs.pop('epochs', 1)
    loss_function = kwargs.pop('loss', nn.CrossEntropyLoss())
    dataloader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer     = kwargs.pop('optimizer', _create_optimizer(model, 'Adam', **kwargs))
    print(f'Using {optimizer.__class__.__name__} optimizer')

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
