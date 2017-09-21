import sys
import time
import numpy as np

from tqdm import tqdm, tqdm_notebook
def _is_in_ipython():
    try: __IPYTHON__; return True
    except NameError: return False
_t = tqdm_notebook if _is_in_ipython() else tqdm


class Stopwatch(object):
    """
    A simple cross-platform
    context-manager stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as s:
    ...     time.sleep(0.1) # doctest: +ELLIPSIS
    Elapsed time: 0.10... sec
    >>> with Stopwatch(verbose=False) as s:
    ...     time.sleep(0.1)
    >>> import math
    >>> math.fabs(s.elapsed() - 0.1) < 0.05
    True
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        if sys.platform == 'win32':
            # on Windows, the best timer is time.clock()
            self._timer_func = time.clock
        else:
            # on most other platforms, the best timer is time.time()
            self._timer_func = time.time
        self.reset()

    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self.verbose:
            print "Elapsed time: {0:.3f} sec".format(self.elapsed())

    def start(self):
        if not self._is_running:
            self._start = self._timer_func()
            self._is_running = True
        return self

    def stop(self):
        if self._is_running:
            self._total += (self._timer_func() - self._start)
            self._is_running = False
        return self

    def elapsed(self):
        if self._is_running:
            now = self._timer_func()
            self._total += (now - self._start)
            self._start = now
        return self._total

    def reset(self):
        self._start = 0.
        self._total = 0.
        self._is_running = False
        return self


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
    if verbose: gen = _t(gen, leave=False, ncols=64, desc='epoch')
    for i in gen:
        yield X[i*batch_size:(i + 1)*batch_size]


def epoch_iter(start_epoch, max_epoch, verbose=False):
    gen = xrange(start_epoch + 1, max_epoch + 1)
    if verbose: gen = _t(gen, leave=True, ncols=84, desc='training')
    for epoch in gen:
        yield epoch


def make_inf_generator(x):
    """Convert to infinite generator.

    Parameters
    ----------
    x : scalar or iterable
        If scalar, always yield that value.
        If finite iterable (e.g. list), first yield all of its values,
            and after that always yield last value.
        If infinite iterable, return generator that yields its values.

    Examples
    --------
    >>> g = make_inf_generator(42)
    >>> for _ in xrange(3): print next(g)
    42
    42
    42
    >>> g2 = make_inf_generator([4, 3, 2])
    >>> for _ in xrange(4): print next(g2)
    4
    3
    2
    2
    >>> def f():
    ...     t = 0
    ...     while True:
    ...         yield t
    ...         t += 1
    >>> g3 = make_inf_generator(f)
    >>> for _ in xrange(3): print next(g3)
    0
    1
    2
    """
    if hasattr(x, '__call__'): # handle generator functions
        x = x()
    if not hasattr(x, '__iter__'):
        x = [x]
    value = None
    for value in x:
        yield value
    while True:
        yield value


if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)
