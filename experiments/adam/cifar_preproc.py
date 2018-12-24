import os
from tqdm import tqdm
import torch
import numpy as np
from ikkuna.utils import load_dataset


dataset_train, dataset_test = load_dataset('CIFAR10')

def zca_whitening_matrix(X):
    '''Compute the ZCA whitening transform.

    Parameters
    ----------
    X   :   np.ndarray
            Data matrix of shape (n_samples, n_features)

    Returns
    -------
    np.ndarray
        (n_features, n_features) whitening matrix. Use it for whitening by X.dot(ZCAMatrix) to
        obtain data with unit variance and zero covariance between features. (Not exactly, but more
        or less)
    '''
    # Compute the covariance
    N         = X.shape[0]
    mean      = X.mean(axis=0)
    sigma     = (X - mean).T.dot((X - mean)) / N
    # Decompose
    U, S, V   = np.linalg.svd(sigma)
    epsilon   = 1e-5
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    return ZCAMatrix


def compute_covariance_matrix(raw_data):
    N        = raw_data.shape[0]
    raw_data = raw_data / raw_data.max()
    mean     = raw_data.mean(axis=0)
    raw_data = raw_data - mean
    cov      = raw_data.T.dot(raw_data) / N
    return cov


if __name__ == '__main__':
    data_train = dataset_train.dataset.data.reshape(50000, -1).astype(np.float)[:3, ...]
    data_test = dataset_test.dataset.data.reshape(10000, -1).astype(np.float)[:3, ...]
    data = np.concatenate([data_train, data_test])
    if False and os.path.exists('cifar_cov.npy'):
        print('Loading covariance matrix')
        cov = np.load('cifar_cov.npy')
    else:
        print('Computing covariance matrix')
        cov = compute_covariance_matrix(data)
        print('Saving covariance matrix')
        np.save('cifar_cov.npy', cov)

    if False and os.path.exists('zca_matrix.npy'):
        print('Loading whitening matrix')
        ZCAMatrix = np.load('zca_matrix.npy')
    else:
        print('Computing whitening transform')
        ZCAMatrix = zca_whitening_matrix(data)
        np.save('zca_matrix.npy', ZCAMatrix)

    print('Whitening data')
    whitened_data = data.dot(ZCAMatrix).astype(np.uint8).reshape(-1, 3, 32, 32)
    whitened_data_train = whitened_data[:3, ...]
    whitened_data_test = whitened_data[3:, ...]
    np.save('whitened_cifar_train.npy', whitened_data_train)
    np.save('whitened_cifar_test.npy', whitened_data_test)

    import matplotlib.pyplot as plt
    orig_img = dataset_train.dataset.data[0]
    plt.figure()
    plt.imshow(orig_img)
    plt.figure()
    plt.imshow(whitened_data_train[0, ...].transpose())
    plt.show()
