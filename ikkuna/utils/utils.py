import torch


def available_optimizers():
    '''List names of all available torch optimizers form :py:mod:`torch.optim`.

    Returns
    -------
    list(str)
        A list of all optimizer names.
    '''
    # get all module properties which aren't magic
    available_optimizers = set(getattr(torch.optim, name) for name in dir(torch.optim) if
                               not name.startswith('__'))
    # remove everything which is a module and not a class (looking at you, lr_scheduler o_o)
    available_optimizers = filter(lambda o: isinstance(o, type), available_optimizers)
    # map them to their class name (w/o module)
    available_optimizers = map(lambda o: o.__name__, available_optimizers)
    return list(available_optimizers)


def create_optimizer(model, name, **kwargs):
    '''Create an optimizer for ``model`` s parameters. Will disregard all params
    with ``requires_grad == False``.

    Parameters
    ----------
    model   :   nn.Module
    name    :   str
                Name of the optimizer
    kwargs  :   dict
                All arguments which should be passed to the optimizer.

    Raises
    ------
    ValueError
        If superflous ``kwargs`` are passed.

    '''

    if name not in available_optimizers():
        raise ValueError(f'Unknown optimizer {name}')

    params = [p for p in model.parameters() if p.requires_grad]
    return getattr(torch.optim, name)(params, **kwargs)


def initialize_model(module, bias_val=0.01):
    '''Perform weight initialization on `module`. This is somewhat hacky since
    it assumes the presence of `weight` and/or `bias` fields on the module. Will
    skip if not present.

    Parameters
    ----------
    module  :   torch.nn.Module
                The model
    bias_val    :   float
                    Constant for biases (should be small and positive)

    Raises
    ------
    ValueError
        If ``module`` is not one of the known models (currently :class:`ikkuna.models.AlexNetMini`
        and :class:`ikkuna.models.DenseNet`)
    '''
    import models
    import torch.nn as nn
    if isinstance(module, models.AlexNetMini):
        for m in module.modules():
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias'):
                module.bias.data.fill_(bias_val)
    elif isinstance(module, models.DenseNet):
        # Official init from torch repo.
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    else:
        raise ValueError(f'Don\'t know how to initialize {module.__class__.__name__}')
