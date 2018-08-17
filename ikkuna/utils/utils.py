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
        and :class:`~models.DenseNet`)
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


####################################################################################################
#                                           NUMBA stuffs                                           #
####################################################################################################

import numba
from numba import cuda
import numpy as np


###################
#  GPU histogram  #
###################
@numba.jit(nopython=True)
def compute_bin(x, n, xmin, xmax):
    # special case to mirror NumPy behavior for last bin
    if x == xmax:
        return n - 1    # a_max always in last bin

    bin = np.int32(n * (x - xmin) / (xmax - xmin))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@cuda.jit
def histogram(x, xmin, xmax, histogram_out):
    nbins = histogram_out.shape[0]

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        # note that calling a numba.jit function from CUDA automatically
        # compiles an equivalent CUDA device function!
        bin_number = compute_bin(x[i], nbins, xmin, xmax)

        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)


@cuda.jit
def min_max(x, min_max_array):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    # Array already seeded with starting values appropriate for x's dtype
    # Not a problem if this array has already been updated
    local_min = min_max_array[0]
    local_max = min_max_array[1]

    for i in range(start, x.shape[0], stride):
        element = x[i]
        local_min = min(element, local_min)
        local_max = max(element, local_max)

    # Now combine each thread local min and max
    cuda.atomic.min(min_max_array, 0, local_min)
    cuda.atomic.max(min_max_array, 1, local_max)


def dtype_min_max(dtype):
    '''Get the min and max value for a numeric dtype'''
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    else:
        info = np.finfo(dtype)
    return info.min, info.max


def numba_gpu_histogram(a_gpu, bins):

    # Find min and max value in array
    dtype_min, dtype_max = dtype_min_max(a_gpu.dtype)
    # Put them in the array in reverse order so that they will be replaced by
    # the first element in the array
    min_max_array_gpu = cuda.to_device(np.array([dtype_max, dtype_min], dtype=a_gpu.dtype))
    min_max[64, 64](a_gpu, min_max_array_gpu)
    a_min, a_max = min_max_array_gpu.copy_to_host()

    # Bin the data into a histogram
    histogram_out = cuda.to_device(np.zeros(shape=(bins,), dtype=np.int32))
    histogram[64, 64](a_gpu, a_min, a_max, histogram_out)

    return histogram_out


##################
#  Tensor2Numba  #
##################
import numba.cuda as cuda
import sys
import torch


def typestr(tensor):
    endianness = '<' if sys.byteorder == 'little' else '>'
    types = {
        torch.float32: 'f4',
        torch.float: 'f4',
        torch.float64: 'f8',
        torch.double: 'f8',
        torch.float16: 'f2',
        torch.half: 'f2',
        torch.uint8: 'u1',
        torch.int8: 'i1',
        torch.int16: 'i2',
        torch.short: 'i2',
        torch.int32: 'i4',
        torch.int: 'i4',
        torch.int64: 'i8',
        torch.long: 'i8'
    }
    return endianness + types[tensor.dtype]


def tensor_to_numba(tensor):
    cai_dict = {
        'shape': tuple(tensor.shape),
        'data': (tensor.data_ptr(), True),
        'typestr': typestr(tensor),
        'version': 0,
        'strides': list(s * tensor.storage().element_size() for s in tensor.stride()),
        'descr': [('', typestr(tensor))]
    }
    setattr(tensor, '__cuda_array_interface__', cai_dict)
    device_array = cuda.as_cuda_array(tensor)
    return device_array
