import numpy as np
from tqdm import tqdm


def batch_iter(X, batch_size=10):
    """Divide input data into batches.

    Examples
    --------
    >>> X = np.arange(36).reshape((12, 3))
    >>> for X_b in batch_iter(X, batch_size=5):
    ...     print X_b
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]
     [12 13 14]]
    [[15 16 17]
     [18 19 20]
     [21 22 23]
     [24 25 26]
     [27 28 29]]
    [[30 31 32]
     [33 34 35]]
    """
    X = np.asarray(X)
    for start_index in range(0, X.shape[0], batch_size):
        yield X[start_index:(start_index + batch_size)]


def tbatch_iter(X, batch_size=10):
    """Same as `batch_iter`, but with progress bar."""
    N = len(X)
    n_batches = N / batch_size + (N % batch_size > 0)
    for X_b in tqdm(batch_iter(X, batch_size=batch_size),
                    total=n_batches, leave=True, ncols=79):
        yield X_b


if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)
