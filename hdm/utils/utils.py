import numpy as np

from tqdm import tqdm, tqdm_notebook
def _is_in_ipython():
    try: __IPYTHON__; return True
    except NameError: return False
progress_bar = tqdm_notebook if _is_in_ipython() else tqdm


def write_during_training(s):
    tqdm.write(s)

def batch_iter(X, batch_size=10, verbose=False):
    """Divide input data into batches, with optional
    progress bar.

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
    N = len(X)
    n_batches = N / batch_size + (N % batch_size > 0)
    gen = range(n_batches)
    if verbose: gen = progress_bar(gen, leave=False, ncols=64, desc='epoch')
    for i in gen:
        yield X[i*batch_size:(i + 1)*batch_size]

def epoch_iter(start_epoch, max_epoch, verbose=False):
    gen = xrange(start_epoch + 1, max_epoch + 1)
    if verbose: gen = progress_bar(gen, leave=True, ncols=84, desc='training')
    for epoch in gen:
        yield epoch

def make_list_from(x):
    return list(x) if hasattr(x, '__iter__') else [x]

def one_hot(y, n_classes=None):
    """Convert `y` to one-hot encoding.

    Examples
    --------
    >>> y = [2, 1, 0, 2, 0]
    >>> one_hot(y)
    array([[ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]])
    """
    n_classes = n_classes or np.max(y) + 1
    return np.eye(n_classes)[y]

def one_hot_decision_function(y):
    """
    Examples
    --------
    >>> y = [[0.1, 0.4, 0.5],
    ...      [0.8, 0.1, 0.1],
    ...      [0.2, 0.2, 0.6],
    ...      [0.3, 0.4, 0.3]]
    >>> one_hot_decision_function(y)
    array([[ 0.,  0.,  1.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.]])
    """
    z = np.zeros_like(y)
    z[np.arange(len(z)), np.argmax(y, axis=1)] = 1
    return z

def unhot(y, n_classes=None):
    """
    Map `y` from one-hot encoding to {0, ..., `n_classes` - 1}.

    Examples
    --------
    >>> y = [[0, 0, 1],
    ...      [0, 1, 0],
    ...      [1, 0, 0],
    ...      [0, 0, 1],
    ...      [1, 0, 0]]
    >>> unhot(y)
    array([2, 1, 0, 2, 0])
    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    if not n_classes:
        _, n_classes = y.shape
    return y.dot(np.arange(n_classes))


if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)
