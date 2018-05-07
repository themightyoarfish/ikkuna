import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam


def _available_optimizers():
    '''List names of all available torch optimizers form :py:mod:`torch.optim`.

    Returns
    -------
    list(str)
    '''
    # get all module properties which aren't magic
    available_optimizers = set(getattr(torch.optim, name) for name in dir(torch.optim) if
                               not name.startswith('__'))
    # remove everything which is a module and not a class (looking at you, lr_scheduler o_o)
    available_optimizers = filter(lambda o: isinstance(o, type), available_optimizers)
    # map them to their class name (w/o module)
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
