import struct
import os.path
import numpy as np


def load_mnist(mode='train', path='.'):
    """
    Load and return MNIST dataset.

    Returns
    -------
    data : (n_samples, 784) np.ndarray
        Data representing raw pixel intensities (in [0., 255.] range).
    target : (n_samples,) np.ndarray
        Labels vector.
    """

    if mode == 'train':
        fname_data = os.path.join(path, 'train-images-idx3-ubyte')
        fname_target = os.path.join(path, 'train-labels-idx1-ubyte')
    elif mode == 'test':
        fname_data = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_target = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("`mode` must be 'test' or 'train'")

    with open(fname_data, 'rb') as fdata:
        magic, n_samples, n_rows, n_cols = struct.unpack(">IIII", fdata.read(16))
        data = np.fromfile(fdata, dtype=np.uint8)
        data = data.reshape(n_samples, n_rows * n_cols)

    with open(fname_target, 'rb') as ftarget:
        magic, n_samples = struct.unpack(">II", ftarget.read(8))
        target = np.fromfile(ftarget, dtype=np.int8)

    return data.astype(float), target


if __name__ == '__main__':
    X, y = load_mnist(mode='train', path='../../data/')
    print X.shape, y.shape