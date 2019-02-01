import struct
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt

from .rng import RNG


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
    dirpath = os.path.join(path, 'mnist/')
    if mode == 'train':
        fname_data = os.path.join(dirpath, 'train-images-idx3-ubyte')
        fname_target = os.path.join(dirpath, 'train-labels-idx1-ubyte')
    elif mode == 'test':
        fname_data = os.path.join(dirpath, 't10k-images-idx3-ubyte')
        fname_target = os.path.join(dirpath, 't10k-labels-idx1-ubyte')
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
        fnames = ['data_batch_{0}'.format(i) for i in range(1, 5 + 1)]
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

def im_flatten(X):
    """Flatten batch of 3-channel images `X`
    for learning.

    Parameters
    ----------
    X : (n_samples, H, W, 3) np.ndarray

    Returns
    -------
    X : (n_samples, H * W * 3) np.ndarray
    """
    X = np.asarray(X)
    if len(X.shape) == 3:
        X = np.expand_dims(X, 0)
    n_samples = X.shape[0]
    X = X.transpose(0, 3, 1, 2).reshape((n_samples, -1))
    if X.shape[0] == 1:
        X = X[0, ...]
    return X

def im_unflatten(X):
    """Convert batch of 3-channel images `X`
    for visualization.

    Parameters
    ----------
    X : (n_samples, D * D * 3) np.ndarray

    Returns
    -------
    X : (n_samples, D, D, 3) np.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 3072)
    >>> Y = X.copy()
    >>> np.testing.assert_allclose(X, im_flatten(im_unflatten(Y)))
    >>> X = np.random.rand(3072)
    >>> Y = X.copy()
    >>> np.testing.assert_allclose(X, im_flatten(im_unflatten(Y)))
    >>> X = np.random.rand(9, 8 * 8 * 3)
    >>> Y = X.copy()
    >>> np.testing.assert_allclose(X, im_flatten(im_unflatten(Y)))
    >>> X = np.random.rand(7, 32, 32, 3)
    >>> Y = X.copy()
    >>> np.testing.assert_allclose(X, im_unflatten(im_flatten(Y)))
    >>> X = np.random.rand(32, 32, 3)
    >>> Y = X.copy()
    >>> np.testing.assert_allclose(X, im_unflatten(im_flatten(Y)))
    >>> X = np.random.rand(8, 8, 3)
    >>> Y = X.copy()
    >>> np.testing.assert_allclose(X, im_unflatten(im_flatten(Y)))
    """
    X = np.asarray(X)
    if len(X.shape) == 1:
        X = np.expand_dims(X, 0)
    D = int(np.sqrt(X.shape[1]/3))
    X = X.reshape((-1, 3, D, D)).transpose(0, 2, 3, 1)
    if X.shape[0] == 1:
        X = X[0, ...]
    return X

def im_rescale(X, mean=0., std=1.):
    """Same as `im_unflatten` but also scale range
    of images for better visual perception.

    Parameters
    ----------
    X : (n_samples, D * D * 3) np.ndarray

    Returns
    -------
    X : (n_samples, D, D, 3) np.ndarray
    """
    X *= std
    X += mean
    X -= X.min(axis=1)[:, np.newaxis]
    X /= X.ptp(axis=1)[:, np.newaxis]  # [0; 1] range for all images
    X = im_unflatten(X)  # (n_samples, D, D, 3)
    X *= 255.
    X = X.astype('uint8')
    return X

def get_cifar10_labels():
    return ['airplane', 'auto', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

def get_cifar10_label(index):
    return get_cifar10_labels()[index]

def plot_cifar10(X, y, samples_per_class=7,
                 title='CIFAR-10 dataset', title_params=None, imshow_params=None):
    # check params
    title_params = title_params or {}
    title_params.setdefault('fontsize', 20)
    title_params.setdefault('y', 0.95)

    imshow_params = imshow_params or {}
    imshow_params.setdefault('interpolation', 'none')

    num_classes = 10
    classes = range(num_classes)
    for c in classes:
        idxs = np.flatnonzero(y == c)
        idxs = RNG(seed=1337).choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + c + 1
            ax = plt.subplot(samples_per_class, num_classes, plt_idx)
            ax.spines['bottom'].set_linewidth(2.)
            ax.spines['top'].set_linewidth(2.)
            ax.spines['left'].set_linewidth(2.)
            ax.spines['right'].set_linewidth(2.)
            plt.tick_params(axis='both', which='both',
                            bottom='off', top='off', left='off', right='off',
                            labelbottom='off', labelleft='off', labelright='off')
            plt.imshow(X[idx].astype('uint8'), **imshow_params)
            if i == 0:
                plt.title(get_cifar10_label(c))
    plt.suptitle(title, **title_params)
    plt.subplots_adjust(wspace=0, hspace=0)


if __name__ == '__main__':
    # run corresponding tests
    from .testing import run_tests
    run_tests(__file__)
