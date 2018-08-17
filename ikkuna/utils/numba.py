'''This module contains functionality for numba-torch interoperability. This isn't used in the
library but may be useful in the future. Documentation is spotty.'''
import numba
import numpy as np
import numba.cuda as cuda
import sys
import torch


@numba.jit(nopython=True)
def compute_bin(x, n, xmin, xmax):
    '''Compute the bins for histogram computation'''
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
    '''Compute a histogram on a numba device array.'''
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
    '''Compute minimum and maximum in parallel on the gpu'''
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
    '''Compute a histogram on a numba device array.'''
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
    '''Convert a tensor to a numba device array. The tensor will share the underlying CUDA storage
    This works by setting the ``__cuda_array_interface__`` attribute on the tensor.

    Parameters
    ----------
    tensor  :   torch.Tensor
    '''
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
