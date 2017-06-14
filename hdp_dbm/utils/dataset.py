import struct
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt

from rng import RNG


def load_mnist(mode='train', path='.'):
    """
    Load and return MNIST dataset.

    Returns
    -------
    data : (n_samples, 784) np.ndarray
        Data representing raw pixel intensities (in [0., 255.] range).
    target : (n_samples,) np.ndarray
        Labels vector (zero-based integers).
    """

    if mode == 'train':
        fname_data = os.path.join(path, 'train-images-idx3-ubyte')
        fname_target = os.path.join(path, 'train-labels-idx1-ubyte')
    elif mode == 'test':
        fname_data = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_target = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("`mode` must be 'train' or 'test'")

    with open(fname_data, 'rb') as fdata:
        magic, n_samples, n_rows, n_cols = struct.unpack(">IIII", fdata.read(16))
        data = np.fromfile(fdata, dtype=np.uint8)
        data = data.reshape(n_samples, n_rows * n_cols)

    with open(fname_target, 'rb') as ftarget:
        magic, n_samples = struct.unpack(">II", ftarget.read(8))
        target = np.fromfile(ftarget, dtype=np.int8)

    return data.astype(float), target


def load_cifar10(mode='train', path='.'):
    """
    Load and return CIFAR-10 dataset.

    Returns
    -------
    data : (n_samples, 3 * 32 * 32) np.ndarray
        Data representing raw pixel intensities (in [0., 255.] range).
    target : (n_samples,) np.ndarray
        Labels vector (zero-based integers).
    """
    dirpath = os.path.join(path, 'cifar-10-batches-py/')
    batch_size = 10000
    if mode == 'train':
        fnames = ['data_batch_{0}'.format(i) for i in xrange(1, 5 + 1)]
    elif mode == 'test':
        fnames = ['test_batch']
    else:
        raise ValueError("`mode` must be 'train' or 'test'")
    n_samples = batch_size * len(fnames)
    data = np.zeros(shape=(n_samples, 3 * 32 * 32), dtype=float)
    target = np.zeros(shape=(n_samples,), dtype=int)
    start = 0
    for fname in fnames:
        fname = os.path.join(dirpath, fname)
        with open(fname, 'rb') as fdata:
            _data = pickle.load(fdata)
            data[start:(start + batch_size)] = np.asarray(_data['data'])
            target[start:(start + batch_size)] = np.asarray(_data['labels'])
        start += 10000
    return data, target


def convert_cifar10(X):
    """
    Convert CIFAR-10 data for visualization.

    Returns
    -------
    data : (n_samples, 32, 32, 3) np.ndarray
    """
    X = X.copy()
    if len(X.shape) == 3:
        X = X.reshape((1, -1))
    return X.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)


def get_cifar10_label(index):
    return {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }[index]


def plot_cifar10(X, y):
    num_classes = 10
    classes = range(num_classes)
    samples_per_class = 7
    for c in classes:
        idxs = np.flatnonzero(y == c)
        idxs = RNG(1337).choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + c + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X[idx].astype('uint8'), interpolation='none')
            plt.axis('off')
            if i == 0:
                plt.title(get_cifar10_label(c))


if __name__ == '__main__':
    X, y = load_cifar10(mode='test', path='../../data/')
    X = convert_cifar10(X)
    import matplotlib.pyplot as plt
    plot_cifar10(X, y)
    plt.show()
