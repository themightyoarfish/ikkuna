import os
from tqdm import tqdm
import torch
import numpy as np
from ikkuna.utils import load_dataset


dataset = load_dataset('CIFAR10')


def compute_covariance_matrix(dataset):
    raw_data = dataset.dataset.data
    N        = raw_data.shape[0]
    raw_data = raw_data / raw_data.max()
    raw_data = raw_data.reshape(N, -1)
    mean     = raw_data.mean(axis=0)
    raw_data = raw_data - mean
    cov      = raw_data.T.dot(raw_data) / (N - 1)
    return cov


if __name__ == '__main__':
    if os.path.exists('cifar_cov.npy'):
        print('Loading covariance matrix')
        cov = np.load('cifar_cov.npy')
    else:
        print('Computing covariance matrix')
        cov = compute_covariance_matrix(dataset[0])
        print('Saving covariance matrix')
        np.save('cifar_cov.npy', cov)

    if os.path.exists('zca_matrix.npy'):
        ZCAMatrix = np.load('zca_matrix.npy')
    else:
        U, S, V   = np.linalg.svd(cov)
        epsilon   = 1e-5
        ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
        np.save('zca_matrix.npy', ZCAMatrix)

    __import__('ipdb').set_trace()
