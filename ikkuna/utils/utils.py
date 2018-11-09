def seed_everything(seed=1234):
    '''Set the seed for :mod:`torch`, :mod:`random`, :mod:`numpy` and :mod:`torch.cuda`, as well as
    the ``PYTHONHASHSEED`` env var. It also configures CuDNN to use deterministic mode.

    .. warning::

        Setting the CuDNN backend to deterministic mode potentially incurs a performance penalty

    Parameters
    ----------
    seed    :   int
                Seed value to use.
    '''
    import random, torch, os, numpy as np       # noqa
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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


def get_model(model_name, *args, **kwargs):
    from ikkuna import models
    try:
        if model_name.startswith('ResNet'):
            model_fn = getattr(models, model_name.lower())
            model = model_fn(*args, **kwargs)
        else:
            Model = getattr(models, model_name)
            model = Model(*args, **kwargs)
    except AttributeError:
        raise ValueError(f'Unknown model {model}')
    else:
        return model


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


import torchvision
from torchvision.transforms import ToTensor, Compose
import numpy as np
import torch


def load_dataset(name, train_transforms=None, test_transforms=None):
    '''Retrieve a dataset and determine the number of classes. This estimate is
    obtained from the number of different values in the training labels.

    Parameters
    ----------
    name    :   str
                Dataset name in :mod:`torchvision.datasets`

    Returns
    -------
    tuple
        2 :class:`~train.DatasetMeta`\ s are returned, one for train and one test set
    '''
    from train import DatasetMeta
    train_transforms = Compose(train_transforms) if train_transforms else ToTensor()
    test_transforms  = Compose(test_transforms) if test_transforms else ToTensor()
    try:
        dataset_cls   = getattr(torchvision.datasets, name)
        dataset_train = dataset_cls('/home/share/data',
                                    download=True,
                                    train=True,
                                    transform=train_transforms
                                    )
        dataset_test  = dataset_cls('/home/share/data',
                                    download=True,
                                    train=False,
                                    transform=test_transforms
                                    )
    except AttributeError:
        raise NameError(f'Dataset {name} unknown.')

    def num_classes(dataset):
        if dataset.train:
            _, labels = dataset.train_data, dataset.train_labels  # noqa
        else:
            _, labels = dataset.test_data, dataset.test_labels  # noqa

        # infer number of classes from labels. will fail if not all classes occur in labels
        if isinstance(labels, np.ndarray):
            return np.unique(labels).size()
        elif isinstance(labels, list):
            return len(set(labels))
        elif isinstance(labels, torch.Tensor):
            return labels.unique().numel()
        else:
            raise ValueError(f'Unexpected label storage {labels.__class__.__name__}')

    def shape(dataset):
        if dataset.train:
            data, _ = dataset.train_data, dataset.train_labels  # noqa
        else:
            data, _ = dataset.test_data, dataset.test_labels  # noqa

        # if only three dimensions, assume [N, H, W], else [N, H, W, C]
        N, H, W = data.shape[:3]
        C = data.shape[-1] if len(data.shape) == 4 else 1
        return (N, H, W, C)

    meta_train = DatasetMeta(dataset=dataset_train, num_classes=num_classes(dataset_train),
                             shape=shape(dataset_train))
    meta_test  = DatasetMeta(dataset=dataset_test, num_classes=num_classes(dataset_test),
                             shape=shape(dataset_test))

    return meta_train, meta_test
