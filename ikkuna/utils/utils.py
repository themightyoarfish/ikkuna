def make_fill_polygons(xs, ys, zs):
    '''Make a set of polygons to fill the space below a line in 3d.

    Returns
    -------
    mpl_toolkits.mplot3d.art3d.Poly3DCollection
    '''
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v = []
    h = 0
    for k in range(0, len(xs) - 1):
        x = [xs[k], xs[k+1], xs[k+1], xs[k]]
        y = [ys[k], ys[k+1], ys[k+1], ys[k]]
        z = [zs[k], zs[k+1],       h,     h]
        v.append(list(zip(x, y, z)))
    poly3dCollection = Poly3DCollection(v)
    return poly3dCollection


def available_optimizers():
    '''List names of all available torch optimizers form :py:mod:`torch.optim`.

    Returns
    -------
    list(str)
        A list of all optimizer names.
    '''
    import torch
    # get all module properties which aren't magic
    available_optimizers = set(getattr(torch.optim, name) for name in dir(torch.optim) if
                               not name.startswith('__'))
    # remove everything which is a module and not a class (looking at you, lr_scheduler o_o)
    available_optimizers = filter(lambda o: isinstance(o, type), available_optimizers)
    # map them to their class name (w/o module)
    available_optimizers = map(lambda o: o.__name__, available_optimizers)
    return list(available_optimizers)


def create_optimizer(model, name, **kwargs):
    '''Create an optimizer for ``model``\ s parameters. Will disregard all params
    with ``requires_grad == False``.

    Parameters
    ----------
    model   :   torch.nn.Module
    name    :   str
                Name of the optimizer
    kwargs  :   dict
                All arguments which should be passed to the optimizer.

    Raises
    ------
    ValueError
        If superfluous ``kwargs`` are passed.

    '''
    import torch
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
        If ``module`` is not one of the known models (currently :class:`~ikkuna.models.AlexNetMini`
        and :class:`~ikkuna.models.DenseNet`)
    '''
    from ikkuna import models
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
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    else:
        raise ValueError(f'Don\'t know how to initialize {module.__class__.__name__}')


import subprocess


def get_memory_stats(mode='total', unit='mb'):
    '''Get the total|used|free gpu memory.

    Parameters
    ----------
    mode    :   str
                One of ``'total'``, ``'used'`` or ``'free'``

    Returns
    -------
    dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    '''
    if mode not in ['total', 'used', 'free']:
        raise ValueError

    units = ['b', 'kb', 'mb', 'gb']
    factors = [1024 ** i for i in range(len(units))]
    if unit not in units:
        raise ValueError(f'Unknown unit "{unit}"')

    memory = subprocess.check_output(
        [
            'nvidia-smi', f'--query-gpu=memory.{mode}',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    index = units.index(unit)
    # nvidia-smi gives mb, so go back to b and then up to the selected unit
    size = [int(x) * factors[2] / factors[index] for x in memory.strip().split('\n')]
    return dict(zip(range(len(size)), size))
