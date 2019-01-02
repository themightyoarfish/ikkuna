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
    import experiments.optimizers
    # get all module properties which aren't magic
    available_optimizers = set(getattr(torch.optim, name)
                               for name in dir(torch.optim)
                               if not name.startswith('__'))
    available_optimizers = available_optimizers.union(set(getattr(experiments.optimizers, name)
                                                          for name in dir(experiments.optimizers)
                                                          if not name.startswith('__')))
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
    import experiments.optimizers
    if name not in available_optimizers():
        raise ValueError(f'Unknown optimizer {name}')

    params = [p for p in model.parameters() if p.requires_grad]
    try:
        return getattr(torch.optim, name)(params, **kwargs)
    except AttributeError:
        return getattr(experiments.optimizers, name)(params, **kwargs)


def get_model(model_name, *args, **kwargs):
    from ikkuna import models
    try:
        if model_name.startswith('ResNet'):
            model_fn = getattr(models, model_name.lower())
            if args:
                print(f'Warning: Ignored args for {model_name} ({args})')
            model = model_fn(**kwargs)
        elif model_name == 'AdamModel':
            from experiments.adam.adam_model import AdamModel
            model = AdamModel(*args, **kwargs)
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


def _load_imagenet_dogs(root, formats, train_transforms, test_transforms):
    '''Load the ImageNetDogs dataset from disk'''
    from torchvision.datasets import DatasetFolder
    from torch.utils.data import random_split
    from PIL import Image
    import copy

    def loader(path):
        '''Load image from fs path'''
        return Image.open(path)

    dataset = DatasetFolder(root, loader, formats)

    size                        = len(dataset)
    size_train                  = int(0.7 * size)
    size_test                   = size - size_train
    dataset_train, dataset_test = random_split(dataset, [size_train, size_test])

    # set the transforms
    # we need to make a copy for the test set to have different transforms
    dataset_test.dataset            = copy.copy(dataset)
    dataset_train.dataset.transform = train_transforms
    dataset_test.dataset.transform  = test_transforms
    return dataset_train, dataset_test


def load_dataset(name, train_transforms=None, test_transforms=None, **kwargs):
    '''Retrieve a dataset and determine the number of classes. This estimate is
    obtained from the number of different values in the training labels.

    Parameters
    ----------
    name    :   str
                Currently, dataset names in :mod:`torchvision.datasets` and ``ImageNetDogs`` are
                supported.
    train_transforms    :   list
                            List of transforms on the train data. Defaults to
                            :class:`torchvision.transforms.ToTensor`
    test_transforms    :   list
                            List of transforms on the test data. Defaults to
                            :class:`torchvision.transforms.ToTensor`

    Keyword Arguments
    -----------------
    root    :   str
                Root directory for dataset folders. Defaults to
                ``/home/share/software/data/<name>/Images``
    formats :   list
                List of file extensions for dataset folders. Defaults to ``['jpg', 'png']``

    Returns
    -------
    tuple
        2 :class:`~train.DatasetMeta`\ s are returned, one for train and one test set
    '''
    def identity(img):
        return img

    train_transforms = Compose(train_transforms) if train_transforms else identity
    test_transforms  = Compose(test_transforms) if test_transforms else identity

    ##########################################
    #  Get the datasets in train/test split  #
    ##########################################
    if name == 'ImageNetDogs':
        root    = kwargs.get('root', f'/home/share/software/data/{name}/Images/')
        formats = kwargs.get('formats', ['jpg', 'png'])
        dataset_train, dataset_test = _load_imagenet_dogs(root, formats, train_transforms,
                                                          test_transforms)
    else:
        try:
            if name == 'WhitenedCIFAR10':
                from experiments.adam.cifar_dataset import WhitenedCIFAR10
                dataset_cls = WhitenedCIFAR10
            else:
                dataset_cls   = getattr(torchvision.datasets, name)

            import os
            if os.path.exists('/home/share/'):
                path = '/home/share/data'
            else:
                path = '/tmp/data'
            dataset_train = dataset_cls(path,
                                        download=True,
                                        train=True,
                                        transform=train_transforms
                                        )
            dataset_test  = dataset_cls(path,
                                        download=True,
                                        train=False,
                                        transform=test_transforms
                                        )
        except AttributeError:
            raise NameError(f'Dataset {name} unknown.')

    ########################
    #  Determine metadata  #
    ########################
    def num_classes(dataset):
        if hasattr(dataset, 'dataset'):     # is a Subset
            dataset = dataset.dataset
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            raise RuntimeError(f'{dataset} has neither `targets` nor `labels` properties.')


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
        N = len(dataset)
        sample, _ = next(iter(dataset))

        # if only three dimensions, assume [N, H, W], else [N, H, W, C]
        if sample.ndimension() == 3:
            C, H, W = sample.shape
        else:
            H, W = sample.shape
            C = 1
        return (N, H, W, C)

    from train import DatasetMeta
    meta_train = DatasetMeta(dataset=dataset_train, num_classes=num_classes(dataset_train),
                             shape=shape(dataset_train))
    meta_test  = DatasetMeta(dataset=dataset_test, num_classes=num_classes(dataset_test),
                             shape=shape(dataset_test))

    return meta_train, meta_test
